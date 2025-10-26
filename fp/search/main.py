import logging
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from constants import BattleType
from fp.battle import Battle
from config import FoulPlayConfig
from .standard_battles import prepare_battles
from .random_battles import prepare_random_battles

from poke_engine import State as PokeEngineState, monte_carlo_tree_search, MctsResult

from fp.search.poke_engine_helpers import battle_to_poke_engine_state

logger = logging.getLogger(__name__)


def select_move_from_mcts_results(mcts_results: list[(MctsResult, float, int)]) -> str:
    final_policy = {}
    for mcts_result, sample_chance, index in mcts_results:
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        logger.info(
            "Policy {}: {} visited {}% avg_score={} sample_chance_multiplier={}".format(
                index,
                this_policy.move_choice,
                round(100 * this_policy.visits / mcts_result.total_visits, 2),
                round(this_policy.total_score / this_policy.visits, 3),
                round(sample_chance, 3),
            )
        )
        for s1_option in mcts_result.side_one:
            final_policy[s1_option.move_choice] = final_policy.get(
                s1_option.move_choice, 0
            ) + (sample_chance * (s1_option.visits / mcts_result.total_visits))

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

    # Select the top move
    logger.info("Top Choice:")
    logger.info(f"\t{round(final_policy[0][1] * 100, 3)}%: {final_policy[0][0]}")

    return final_policy[0][0]


def get_result_from_mcts(state: str, search_time_ms: int, index: int) -> MctsResult:
    logger.debug("Calling with {} state: {}".format(index, state))
    poke_engine_state = PokeEngineState.from_string(state)

    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)
    logger.info("Iterations {}: {}".format(index, res.total_visits))
    return res


def search_time_num_battles_randombattles(battle):
    revealed_pkmn = len(battle.opponent.reserve)
    if battle.opponent.active is not None:
        revealed_pkmn += 1

    opponent_active_num_moves = len(battle.opponent.active.moves)
    
    # Calculate default configuration
    default_num_battles_multiplier = 6
    search_time_per_battle = FoulPlayConfig.search_time_ms
    
    # Calculate time pressure threshold
    time_limit = ((search_time_per_battle / 1000) * default_num_battles_multiplier) * 2 + 15
    in_time_pressure = (
        battle.time_remaining is not None and battle.time_remaining <= time_limit
    )

    # Determine number of battles based on game state
    if opponent_active_num_moves == 0:
        # No moves shown: default to 60 battles
        num_battles_multiplier = 1 if in_time_pressure else default_num_battles_multiplier
    elif revealed_pkmn <= 1:
        # 1 move shown but only 1 Pokémon revealed: 60 battles
        num_battles_multiplier = 1 if in_time_pressure else default_num_battles_multiplier
    else:
        # 2-6 Pokémon revealed: scale down proportionally
        # 2 revealed -> 60, 3 -> 50, 4 -> 40, 5 -> 30, 6 -> 20
        scaled_multiplier = int(((7 - revealed_pkmn) / 5) * default_num_battles_multiplier)
        # Round up to nearest 10, then divide by parallelism
        scaled_multiplier = max(2, ((scaled_multiplier * FoulPlayConfig.parallelism + 9) // 10) * 10 // FoulPlayConfig.parallelism)
        num_battles_multiplier = 1 if in_time_pressure else scaled_multiplier

    return FoulPlayConfig.parallelism * num_battles_multiplier, search_time_per_battle


def search_time_num_battles_standard_battle(battle):
    opponent_active_num_moves = len(battle.opponent.active.moves)

    default_num_battles_multiplier = 6
    time_limit = ((FoulPlayConfig.search_time_ms / 1000) * default_num_battles_multiplier) * 2 + 15

    in_time_pressure = (
        battle.time_remaining is not None and battle.time_remaining <= time_limit
    )

    if (
        battle.team_preview
        or (battle.opponent.active.hp > 0 and opponent_active_num_moves == 0)
        or opponent_active_num_moves < 3
    ):
        num_battles_multiplier = 1 if in_time_pressure else default_num_battles_multiplier
        return FoulPlayConfig.parallelism * num_battles_multiplier, int(
            FoulPlayConfig.search_time_ms
        )
    else:
        return FoulPlayConfig.parallelism, FoulPlayConfig.search_time_ms


def find_best_move(battle: Battle) -> str:
    battle = deepcopy(battle)
    if battle.team_preview:
        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)

    if battle.battle_type == BattleType.RANDOM_BATTLE:
        num_battles, search_time_per_battle = search_time_num_battles_randombattles(
            battle
        )
        battles = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.BATTLE_FACTORY:
        num_battles, search_time_per_battle = search_time_num_battles_standard_battle(
            battle
        )
        battles = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.STANDARD_BATTLE:
        num_battles, search_time_per_battle = search_time_num_battles_standard_battle(
            battle
        )
        battles = prepare_battles(battle, num_battles)
    else:
        raise ValueError("Unsupported battle type: {}".format(battle.battle_type))

    logger.info("Searching for a move using MCTS...")
    logger.info(
        "Sampling {} battles at {}ms each".format(num_battles, search_time_per_battle)
    )
    logger.info(f"Time remaining: {battle.time_remaining}")
    with ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism) as executor:
        futures = []
        for index, (b, chance) in enumerate(battles):
            fut = executor.submit(
                get_result_from_mcts,
                battle_to_poke_engine_state(b).to_string(),
                search_time_per_battle,
                index,
            )
            futures.append((fut, chance, index))

    mcts_results = [(fut.result(), chance, index) for (fut, chance, index) in futures]
    choice = select_move_from_mcts_results(mcts_results)
    logger.info("Choice: {}".format(choice))
    return choice
