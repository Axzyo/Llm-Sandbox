import argparse
import os
import queue
import random
import threading
import time

import pygame

from sim.actions import DIRS, attempt_move, evaluate_interact
from sim.brain import Brain
from sim.config import load_config
from sim.engine import Engine, broadcast, reap_dead
from sim.entities import Entity
from sim.goals import goal_from_intent
from sim.journal import Journal
from sim.terrain import HEIGHT, SPAWNS, WIDTH, build_test_map, place_resources
from sim.provider import OllamaProvider
from sim.world import chebyshev

TILE_SIZE = 32

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

AUTOTEST_MESSAGE = "hello, can anyone hear me?"
AUTOTEST_TIMEOUT_S = 30.0

think_q: queue.Queue = queue.Queue()
result_q: queue.Queue = queue.Queue()


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
    for e in [player, *npcs]:
        world.entities[e.id] = e
        e.properties["interact_range"] = int(cfg["interact_range"])
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
    engine = Engine(world, npcs, brains, journal,
                    dispatch_think=lambda eid, snap, evs: think_q.put(("think", eid, snap, evs)))
    engine.track(player)
    npcs_by_id = engine.npcs_by_id

    hear_log = engine.hear_log
    live_text: dict = {}          # eid -> say text streaming in right now (display only)
    live_lock = threading.Lock()
    worker = threading.Thread(target=think_worker, args=(brains, live_text, live_lock), daemon=True)
    worker.start()
    # Load the model now so the first think isn't a multi-second cold start.
    threading.Thread(target=provider.warm, daemon=True).start()

    input_buffer = ""
    composing = False
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

        # deliver finished thinks first so their goals enact this same frame
        while True:
            try:
                kind, eid, goals = result_q.get_nowait()
            except queue.Empty:
                break
            engine.post_goals(eid, goals)
            with live_lock:
                live_text.pop(eid, None)  # a streamed say line broadcasts when its goal enacts

        engine.step(dt)
        sim_t = engine.sim_t

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
                        broadcast(world, player, text, brains, engine.pending_obs, sim_t, hear_log, journal)
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
            player.next_move_at = sim_t + player.properties["move_interval"]
            if result["ok"]:
                last_fail = None
                journal.log("player", "action_complete", action="move", **result)
            else:
                sig = (held, result["reason"], result.get("by"))
                if sig != last_fail:
                    last_fail = sig
                    journal.log("player", "action_failed", action="move", **result)

        if autotest and outcome is None:
            listener = "npc_1" if args.autotest == "talk" else "npc_2"
            heard = [m for m in brains[listener].store.memories if m["sense"] == "heard"]
            if heard:
                said = (heard[-1].get("subject") or {}).get("info", {}).get("text", "")
                outcome = f"PASS - {listener} heard: {said}"
                print(f"AUTOTEST {outcome}", flush=True)
                journal.log("system", "autotest_result", result=outcome)
                running = False

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
            if spk is not None and chebyshev(player.x, player.y, spk.x, spk.y) <= player.properties["hearing_radius"]:
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

        if inspect_npc_id is not None and inspect_npc_id not in npcs_by_id:
            inspect_npc_id = None          # the inspected NPC died: close its panel
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
