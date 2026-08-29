import argparse
import json
import os
import queue
import random
import threading
import time

import pygame

from sim.actions import DIRS, attempt_move, evaluate_interact
from sim.brain import Brain
from sim.entities import Entity
from sim.goals import goal_from_intent
from sim.journal import Journal
from sim.terrain import HEIGHT, SPAWNS, WIDTH, build_test_map, place_resources, tick_resources, interact_with, use_item
from sim.pathing import next_step
from sim.needs import tick_needs, sensations as needs_sensations
from sim.perception import PerceptionTracker, visible_entities, visible_tiles
from sim.provider import OllamaProvider
from sim.world import chebyshev

TILE_SIZE = 32
PERCEPTION_INTERVAL = 0.2

KEY_DIRS = [
    (pygame.K_e, "up"),
    (pygame.K_s, "left"),
    (pygame.K_d, "down"),
    (pygame.K_f, "right"),
]

COLOR_FLOOR = (38, 42, 50)
COLOR_WALL = (96, 104, 118)
COLOR_GRID = (48, 53, 63)
COLOR_PLAYER = (90, 180, 255)
COLOR_NPC = (255, 170, 70)
COLOR_TEXT = (225, 228, 235)
COLOR_CHAT = (255, 236, 180)
COLOR_TARGET = (255, 240, 120)

ENABLE_NPC_MEMORY = True
ENABLE_NPC_THINKS = True

AUTOTEST_MESSAGE = "hello, can anyone hear me?"
AUTOTEST_TIMEOUT_S = 30.0

think_q: queue.Queue = queue.Queue()
result_q: queue.Queue = queue.Queue()


def load_config() -> dict:
    cfg = {
        "ollama_url": "http://localhost:11434",
        "model": "gemma4",
        "temperature": 0.2,
        "num_predict": 400,
        "keep_alive": "30m",
        "memory_k": 5,
        "memory_halflife_s": 300.0,
        "interact_range": 4,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    return cfg


def build_snapshot(npc, world, pending_events: list, now_t: float) -> dict:
    return {
        "t": round(now_t, 2),
        "self_id": npc.id,
        "self_pos": [npc.x, npc.y],
        "health": round(npc.hp),                # three survival needs, 0 (empty) .. 100 (full)
        "hunger": round(npc.hunger),
        "thirst": round(npc.thirst),
        "sensations": needs_sensations(npc),    # how those needs feel right now
        "drives": npc.drives,                    # personality weights (survival, curiosity, ...)
        "vision_radius": npc.vision_radius,
        "hearing_radius": npc.hearing_radius,
        "interact_range": npc.interact_range,
        "visible_entities": [{"id": e.id, "type": e.kind, "pos": [e.x, e.y]} for e in visible_entities(world, npc)],
        "recent_perceptions": pending_events[-12:],
    }


def remember_action(brains: dict, npc, event: dict, sim_t: float) -> None:
    """Write a 'did' memory of a resolved action outcome, through the same pipeline."""
    if ENABLE_NPC_MEMORY and npc.id in brains:
        brains[npc.id].record_events([event], sim_t, location=[npc.x, npc.y])


def broadcast(world, speaker, text: str, brains: dict, pending_obs: dict, sim_t: float,
              hear_log: list, journal: Journal) -> None:
    """Speech is a broadcast: the speaker remembers saying it, and every entity within
    its own hearing_radius forms a `heard` memory (walls don't block sound). The player,
    if in earshot, just sees it in the log."""
    journal.log(speaker.id, "say", text=text)
    print(f"[{sim_t:7.1f}] {'you' if speaker.id == 'player' else speaker.id}: {text}", flush=True)
    if ENABLE_NPC_MEMORY and speaker.id in brains:
        remember_action(brains, speaker, {"kind": "did_say", "text": text}, sim_t)
    for e in world.entities.values():
        if e.id == speaker.id:
            continue
        if chebyshev(speaker.x, speaker.y, e.x, e.y) > e.hearing_radius:
            continue
        event = {"kind": "heard_say", "speaker": speaker.id, "speaker_type": speaker.kind,
                 "speaker_pos": [speaker.x, speaker.y], "text": text}
        if ENABLE_NPC_MEMORY and e.id in brains:
            brains[e.id].record_events([event], sim_t, location=[e.x, e.y])
            pending_obs[e.id].append(event)
        elif e.id == "player":
            hear_log.append(f"{speaker.id}: {text}")


def enact_instant(world, npc, action_obj, journal: Journal, sim_t: float, brains: dict,
                  pending_obs: dict, hear_log: list) -> str:
    """Enact a one-tick action (interact / inventory / say). Returns 'done' or 'failed'."""
    action = action_obj.get("action")
    params = action_obj.get("params") or {}
    if action == "interact":
        res = evaluate_interact(world, npc, params.get("target"))
        target = world.entities.get(res.get("target"))
        tpos = [target.x, target.y] if target is not None else None
        ttype = target.kind if target is not None else "entity"
        if res["ok"]:
            journal.log(npc.id, "action_complete", action="interact", **res)
            # a resource target's data decides what interacting does (drink / pick)
            eff = interact_with(target, npc, sim_t) if target is not None else None
            effect = None
            if eff is not None:
                journal.log(npc.id, "resource_use", target=target.id, **eff)
                if eff["did"] == "drink":
                    effect = f"{eff['stat']} +{eff['gained']:g}"
                elif eff["did"] == "harvest":
                    effect = f"picked a {eff['yields']}" if eff["ok"] else "nothing to pick"
            remember_action(brains, npc, {"kind": "did_interact", "target": res.get("target"),
                                          "target_type": ttype, "target_pos": tpos,
                                          "outcome": "ok", "effect": effect}, sim_t)
            return "done"
        journal.log(npc.id, "action_failed", action="interact", **res)
        outcome = "out_of_range" if not res.get("range_ok") else ("no_los" if not res.get("los_ok") else "failed")
        remember_action(brains, npc, {"kind": "did_interact", "target": params.get("target"),
                                      "target_type": ttype, "target_pos": tpos, "outcome": outcome}, sim_t)
        return "failed"
    if action == "inventory":
        op, item = params.get("op"), params.get("item")
        if item in npc.inventory:
            # 'use' applies the item's own effect (eat a berry -> hunger); the item
            # is consumed. drop/arrange have no effect yet.
            used = use_item(npc, item) if op == "use" else None
            journal.log(npc.id, "action_complete", action="inventory", op=op, item=item,
                        **(used or {}))
            effect = f"{used['stat']} +{used['gained']:g}" if used else None
            remember_action(brains, npc, {"kind": "did_inventory", "op": op, "item": item,
                                          "outcome": "ok", "effect": effect}, sim_t)
            return "done"
        journal.log(npc.id, "action_failed", action="inventory", op=op, item=item, reason="no_such_item")
        remember_action(brains, npc, {"kind": "did_inventory", "op": op, "item": item, "outcome": "no_such_item"}, sim_t)
        return "failed"
    if action == "say":
        broadcast(world, npc, params.get("text", ""), brains, pending_obs, sim_t, hear_log, journal)
        return "done"
    # wait/recall are thinking-layer choices and never become goals, so they don't reach here
    journal.log(npc.id, "action_failed", action=str(action), reason="unknown_action")
    return "failed"


def progress_move(world, npc, goal, action_obj, journal: Journal, sim_t: float, brains: dict) -> str:
    """Advance a durative move action one step. Returns 'active' (more to go),
    'done' (arrived) or 'failed' (unreachable)."""
    params = action_obj.get("params") or {}
    tx, ty = params.get("x"), params.get("y")
    if (npc.x, npc.y) == (tx, ty):
        remember_action(brains, npc, {"kind": "did_move", "pos": [tx, ty], "outcome": "arrived"}, sim_t)
        return "done"
    if goal.started_step != goal.step:
        goal.started_step = goal.step
        journal.log(npc.id, "action_start", action="move", to=[tx, ty], source="goal")
    if sim_t < npc.next_move_at:
        return "active"
    step = next_step(world, (npc.x, npc.y), (tx, ty))
    if step is None:
        journal.log(npc.id, "action_failed", action="move", reason="unreachable", to=[tx, ty])
        remember_action(brains, npc, {"kind": "did_move", "pos": [tx, ty], "outcome": "unreachable"}, sim_t)
        return "failed"
    result = attempt_move(world, npc, step[0] - npc.x, step[1] - npc.y)
    npc.next_move_at = sim_t + npc.move_interval
    if result["ok"]:
        journal.log(npc.id, "action_complete", action="move", **result)
    # blocked/occupied: stay active and retry next tick (pathing reroutes)
    return "active"


def advance_goals(world, npc, journal: Journal, sim_t: float, brains: dict,
                  pending_obs: dict, hear_log: list) -> None:
    """Work the NPC's top goal for this tick. Durative goals stay 'active' and
    resume next tick unless a higher-importance goal has since preempted them."""
    goal = npc.goals.current()
    if goal is None:
        return
    action_obj = goal.current_action
    if action_obj is None:                 # plan exhausted -> the goal is done
        npc.goals.complete(goal)
        return
    goal.status = "active"
    if action_obj.get("action") == "move":
        outcome = progress_move(world, npc, goal, action_obj, journal, sim_t, brains)
    else:
        outcome = enact_instant(world, npc, action_obj, journal, sim_t, brains, pending_obs, hear_log)
    if outcome == "done":
        if not goal.advance():             # no actions left -> whole plan complete
            npc.goals.complete(goal)
    elif outcome == "failed":
        npc.goals.fail(goal)               # a failed step abandons the whole plan
    # "active": leave it in place for next tick


def reap_dead(npcs: list, npcs_by_id: dict, world, journal: Journal, sim_t: float) -> list:
    """Remove NPCs whose health has hit 0 — death is permanent, there is no respawn.
    A dead NPC leaves the world (others perceive it disappear), stops thinking and
    perceiving, and its death is logged. Returns the ones that died this tick."""
    dead = [n for n in npcs if n.hp <= 0.0]
    for npc in dead:
        cause = "dehydration" if npc.thirst <= 0.0 else ("starvation" if npc.hunger <= 0.0 else "unknown")
        journal.log(npc.id, "death", pos=[npc.x, npc.y], cause=cause)
        print(f"[{sim_t:7.1f}] {npc.id} died of {cause}", flush=True)
        world.entities.pop(npc.id, None)
        npcs_by_id.pop(npc.id, None)
        npcs.remove(npc)
    return dead


def think_worker(brains: dict, live_text: dict, live_lock: threading.Lock) -> None:
    while True:
        job = think_q.get()
        if job is None:
            return
        kind, eid, payload, events = job
        with live_lock:
            live_text[eid] = ""

        def on_delta(delta: str, _eid=eid):
            with live_lock:
                live_text[_eid] = live_text.get(_eid, "") + delta

        # stream the think so a `say` types out live; the broadcast still fires only
        # on completion, in enact_instant (you don't "hear" half a sentence).
        # Bridge: the brain returns either a list of goals (multi-goal brain) or a
        # single validated intent dict (current brain) -> normalize to a goal list.
        out = brains[eid].decide(payload, events, on_delta=on_delta)
        if isinstance(out, list):
            goals = out
        else:
            g = goal_from_intent(out)
            goals = [g] if g is not None else []
        result_q.put((kind, eid, goals))


def build_autotest_script() -> list:
    # ENTER opens compose, type the message, ENTER broadcasts.
    script = [(1.0, pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": "\r"})]
    t = 1.3
    for ch in AUTOTEST_MESSAGE:
        script.append((t, pygame.TEXTINPUT, {"text": ch}))
        t += 0.02
    script.append((t + 0.3, pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": "\r"}))
    return script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--autotest", choices=["talk", "npc-talk"])
    args = parser.parse_args()
    autotest = args.autotest is not None

    cfg = load_config()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * TILE_SIZE, HEIGHT * TILE_SIZE))
    pygame.display.set_caption(f"llm npc sandbox [{cfg['model']}]" + (f" - autotest {args.autotest}" if autotest else ""))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    world, spawns = build_test_map()
    player = Entity("player", "you", "player", *spawns["player"])
    npcs = [
        Entity("npc_1", "npc_1", "npc", *spawns["npc_1"]),
        Entity("npc_2", "npc_2", "npc", *spawns["npc_2"]),
    ]
    npcs_by_id = {}
    for e in [player, *npcs]:
        world.entities[e.id] = e
        npcs_by_id[e.id] = e
        e.interact_range = int(cfg["interact_range"])
    # scatter terrain resources (fresh random layout each load)
    place_resources(world, random.Random())

    run_id = time.strftime("r_%Y%m%d_%H%M%S")
    journal = Journal(os.path.join("runs", f"{run_id}.jsonl"), run_id)
    journal.log("system", "spawn", model=cfg["model"],
                entities={e.id: list(e.pos) for e in world.entities.values()})

    provider = OllamaProvider(
        cfg["ollama_url"], cfg["model"], cfg["temperature"], cfg["num_predict"],
        keep_alive=cfg.get("keep_alive", "30m"),
    )
    brains = {
        npc.id: Brain(npc.id, provider, journal,
                      memory_k=cfg["memory_k"], memory_halflife_s=cfg["memory_halflife_s"])
        for npc in npcs
    }
    trackers = {npc.id: PerceptionTracker(npc.id) for npc in npcs}
    pending_obs = {e.id: [] for e in [player, *npcs]}
    thinking = {npc.id: False for npc in npcs}
    hear_log: list = []
    live_text: dict = {}          # eid -> say text streaming in right now (display only)
    live_lock = threading.Lock()
    worker = threading.Thread(target=think_worker, args=(brains, live_text, live_lock), daemon=True)
    worker.start()
    # Load the model now so the first think isn't a multi-second cold start.
    threading.Thread(target=provider.warm, daemon=True).start()

    input_buffer = ""
    composing = False
    next_perceive_at = 0.0
    sim_t = 0.0
    frame = 0
    inspect_npc_id: str | None = None
    last_fail = None
    running = True
    outcome = None

    script: list = []
    intent_script: list = []
    if args.autotest == "talk":
        occupant = world.entity_at(18, 9)
        assert not world.blocked(18, 9) and (occupant is None or occupant.id == "player")
        player.x, player.y = 18, 9   # within earshot of npc_1 at (19, 9)
        script = build_autotest_script()
        print(f"AUTOTEST talk: player at {(player.x, player.y)}, npc_1 at {spawns['npc_1']}", flush=True)
    elif args.autotest == "npc-talk":
        intent_script = [(1.0, "npc_1", {"action": "say", "params": {"text": "is anyone out there?"}})]
        print(f"AUTOTEST npc-talk: npc_1 at {spawns['npc_1']}, npc_2 at {spawns['npc_2']}", flush=True)

    while running:
        dt = clock.tick(60) / 1000.0
        sim_t += dt

        # survival needs drain in real time -> the pressure NPCs perceive and weigh
        for npc in npcs:
            tick_needs(npc, dt)
        tick_resources(world, sim_t)   # regrow berries whose timer elapsed
        reap_dead(npcs, npcs_by_id, world, journal, sim_t)   # 0 hp is death, no respawn

        while script and sim_t >= script[0][0]:
            _, etype, data = script.pop(0)
            pygame.event.post(pygame.event.Event(etype, data))

        while intent_script and sim_t >= intent_script[0][0]:
            _, actor_id, injected_intent = intent_script.pop(0)
            g = goal_from_intent(injected_intent, importance=10.0)
            result_q.put(("think", actor_id, [g] if g is not None else []))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if composing:
                    composing = False
                    input_buffer = ""
                else:
                    running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                tx, ty = ev.pos[0] // TILE_SIZE, ev.pos[1] // TILE_SIZE
                ent = world.entity_at(tx, ty)
                if ent is not None and ent.kind == "npc":
                    inspect_npc_id = None if inspect_npc_id == ent.id else ent.id
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                if not composing:
                    composing = True
                    input_buffer = ""
                else:
                    text = input_buffer.strip()
                    composing = False
                    input_buffer = ""
                    if text:
                        broadcast(world, player, text, brains, pending_obs, sim_t, hear_log, journal)
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_BACKSPACE:
                if composing:
                    input_buffer = input_buffer[:-1]
            elif ev.type == pygame.TEXTINPUT:
                if composing:
                    input_buffer += ev.text

        keys = pygame.key.get_pressed()
        mx, my = pygame.mouse.get_pos()
        hovered = world.entity_at(mx // TILE_SIZE, my // TILE_SIZE)
        player.target = hovered.id if hovered is not None and hovered.id != "player" else None

        held = next((name for key, name in KEY_DIRS if keys[key]), None)
        if held and not composing and sim_t >= player.next_move_at:
            dx, dy = DIRS[held]
            result = attempt_move(world, player, dx, dy)
            player.next_move_at = sim_t + player.move_interval
            if result["ok"]:
                last_fail = None
                journal.log("player", "action_complete", action="move", **result)
            else:
                sig = (held, result["reason"], result.get("by"))
                if sig != last_fail:
                    last_fail = sig
                    journal.log("player", "action_failed", action="move", **result)

        if sim_t >= next_perceive_at:
            next_perceive_at = sim_t + PERCEPTION_INTERVAL
            for npc in npcs:
                if ENABLE_NPC_MEMORY:
                    # geometry perception -> spatial memory (a remembered map), separate
                    # from episodic memory so ~289 seen tiles never swamp recall. Then a
                    # maintenance pass decays memorability and forgets faded tiles, keeping
                    # geometry near current goal locations.
                    brains[npc.id].perceive_tiles(visible_tiles(world, npc))
                    brains[npc.id].maintain_spatial(npc.goals.locations())
                events = trackers[npc.id].update(world, npc)
                if events:
                    journal.log(npc.id, "perception", events=events)
                    if ENABLE_NPC_MEMORY:
                        brains[npc.id].record_events(events, sim_t, location=[npc.x, npc.y])
                    pending_obs[npc.id].extend(events)
                    for ev_ in events:
                        if ev_["kind"] in ("entity_entered", "entity_moved"):
                            npc.target = ev_["id"]
                        elif ev_["kind"] == "entity_left" and npc.target == ev_["id"]:
                            npc.target = None

        for npc in npcs:
            if not ENABLE_NPC_THINKS or thinking[npc.id]:
                continue
            if not brains[npc.id].pending_think:   # think only when something novel happened
                continue
            brains[npc.id].pending_think = False
            events = list(pending_obs[npc.id])
            pending_obs[npc.id] = []
            snapshot = build_snapshot(npc, world, events, sim_t)
            thinking[npc.id] = True
            journal.log(npc.id, "action_start", action="think")
            think_q.put(("think", npc.id, snapshot, events))

        while True:
            try:
                kind, eid, goals = result_q.get_nowait()
            except queue.Empty:
                break
            thinking[eid] = False
            if goals and eid in npcs_by_id:          # eid may have died while thinking
                npcs_by_id[eid].goals.add_many(goals)  # merged + re-sorted by importance
                journal.log(eid, "goals_added", goals=[g.summary() for g in goals],
                            importances=[g.importance for g in goals])
            with live_lock:
                live_text.pop(eid, None)  # a streamed say line broadcasts when its goal enacts

        if autotest and outcome is None:
            listener = "npc_1" if args.autotest == "talk" else "npc_2"
            heard = [m for m in brains[listener].store.memories if m["sense"] == "heard"]
            if heard:
                said = (heard[-1].get("subject") or {}).get("info", {}).get("text", "")
                outcome = f"PASS - {listener} heard: {said}"
                print(f"AUTOTEST {outcome}", flush=True)
                journal.log("system", "autotest_result", result=outcome)
                running = False

        for npc in npcs:
            advance_goals(world, npc, journal, sim_t, brains, pending_obs, hear_log)

        if autotest and outcome is None and sim_t > AUTOTEST_TIMEOUT_S:
            outcome = f"FAIL - timeout ({args.autotest})"
            print(f"AUTOTEST {outcome}", flush=True)
            journal.log("system", "autotest_result", result=outcome)
            running = False

        screen.fill(COLOR_FLOOR)
        for ty in range(HEIGHT):
            for tx in range(WIDTH):
                rect = pygame.Rect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if world.is_wall(tx, ty):
                    pygame.draw.rect(screen, COLOR_WALL, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FLOOR, rect)
                    pygame.draw.rect(screen, COLOR_GRID, rect, 1)
        for e in world.entities.values():
            rect = pygame.Rect(e.x * TILE_SIZE + 4, e.y * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8)
            color = COLOR_PLAYER if e.kind == "player" else COLOR_NPC
            pygame.draw.rect(screen, color, rect, border_radius=6)
            if e.id == player.target and e.kind != "player":
                pygame.draw.rect(screen, COLOR_TARGET, rect.inflate(6, 6), 2, border_radius=8)
            label = font.render(e.name, True, COLOR_TEXT)
            screen.blit(label, (e.x * TILE_SIZE + 2, (e.y + 1) * TILE_SIZE - 12))

        target_info = "target: none"
        if player.target is not None:
            tgt = world.entities.get(player.target)
            chk = evaluate_interact(world, player, player.target)
            target_info = (
                f"target: {tgt.id} | dist:{chk['distance']} "
                f"range:{'ok' if chk['range_ok'] else 'FAIL'} los:{'ok' if chk['los_ok'] else 'BLOCKED'}"
            )

        visible_hud: list = ["esdf move | enter: speak | ctrl+click npc: memories | esc quit", target_info]
        for line in hear_log[-5:]:
            visible_hud.append(line)
        with live_lock:
            live_now = {eid: t for eid, t in live_text.items() if t}
        for eid, partial in live_now.items():   # an NPC mid-sentence, if the player can hear it
            spk = world.entities.get(eid)
            if spk is not None and chebyshev(player.x, player.y, spk.x, spk.y) <= player.hearing_radius:
                visible_hud.append(f"{eid}: {partial}█")
        if composing:
            visible_hud.append(f"> {input_buffer}_")
        screen_panel = pygame.Surface((480, 26 + 18 * len(visible_hud)), pygame.SRCALPHA)
        screen_panel.fill((10, 12, 16, 190))
        screen.blit(screen_panel, (6, 6))
        for i, line in enumerate(visible_hud):
            color = COLOR_CHAT if 1 < i < 2 + len(hear_log[-5:]) else COLOR_TEXT
            screen.blit(font.render(line, True, color), (12, 8 + i * 18))

        statuses = []
        for npc in npcs:
            st = f"watching:{npc.target}" if npc.target is not None else "idle"
            g = npc.goals.current()
            gtxt = f"{g.summary()} i={g.importance:g} ({len(npc.goals)})" if g is not None else "no-goal"
            statuses.append(f"{npc.id}: {st} | {gtxt}")
        bar = font.render(" | ".join(statuses), True, COLOR_TEXT)
        screen.blit(bar, (10, HEIGHT * TILE_SIZE - 22))

        if inspect_npc_id is not None and inspect_npc_id in brains:
            store = brains[inspect_npc_id].store
            mems = sorted(list(store.memories), key=lambda m: m["t"], reverse=True)[:20]
            panel_w = 340
            insp_npc = npcs_by_id[inspect_npc_id]
            lines = [f"{inspect_npc_id} goals ({len(insp_npc.goals)}) - by importance"]
            if len(insp_npc.goals) == 0:
                lines.append("(none)")
            for gi in insp_npc.goals:
                lines.append(f"  i={gi.importance:<4g} {gi.summary()} [{gi.status}]")
            lines.append(f"{inspect_npc_id} memories ({len(store)}) - ctrl+click to close")
            lines.append(f"{'time':>6} {'sense':<5} {'target':<9} {'dir':<3} {'mypos'}")
            if not mems:
                lines.append("(none yet)")
            for m in mems:
                s = m.get("subject") or {}
                target = s.get("ref") or s.get("type") or "-"
                span = f" x{m['count']}@{m['t_end']:.1f}" if m.get("count", 1) > 1 else ""
                lines.append(
                    f"{m['t']:6.1f} {m['sense']:<5} {str(target):<9} "
                    f"{(m.get('direction') or '-'):<3} {m.get('observer_loc')}{span}"
                )
            panel_x = WIDTH * TILE_SIZE - panel_w - 6
            panel = pygame.Surface((panel_w, 20 + 18 * len(lines)), pygame.SRCALPHA)
            panel.fill((10, 12, 16, 210))
            screen.blit(panel, (panel_x, 6))
            for i, line in enumerate(lines):
                color = COLOR_TARGET if i == 0 else COLOR_TEXT
                screen.blit(font.render(line, True, color), (panel_x + 8, 10 + i * 18))

        pygame.display.flip()
        frame += 1
        if args.frames and frame >= args.frames:
            running = False

    think_q.put(None)
    journal.log("system", "shutdown", frames=frame)
    journal.close()
    pygame.quit()


if __name__ == "__main__":
    main()
