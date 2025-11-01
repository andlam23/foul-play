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


def select_move_from_mcts_results(mcts_results: list) -> str:
    """
    Select a move using pure weighted random selection from MCTS results.
    All moves are candidates - you pick based on their aggregated scores.
    """
    final_policy = {}
    
    # Print header
    print("\n" + "="*80)
    print("MCTS SEARCH RESULTS".center(80))
    print("="*80 + "\n")
    
    # Print individual policy results
    for mcts_result, sample_chance, index in mcts_results:
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        visit_pct = round(100 * this_policy.visits / mcts_result.total_visits, 2)
        avg_score = round(this_policy.total_score / this_policy.visits, 3)
        sample_mult = round(sample_chance, 3)
        
        print(f"Policy #{index}:")
        print(f"  Move: {this_policy.move_choice}")
        print(f"  Visits: {visit_pct}% | Avg Score: {avg_score} | Sample Weight: {sample_mult}")
        print()
        
        logger.info(
            "Policy {}: {} visited {}% avg_score={} sample_chance_multiplier={}".format(
                index,
                this_policy.move_choice,
                visit_pct,
                avg_score,
                sample_mult,
            )
        )
        
        for s1_option in mcts_result.side_one:
            final_policy[s1_option.move_choice] = final_policy.get(
                s1_option.move_choice, 0
            ) + (sample_chance * (s1_option.visits / mcts_result.total_visits))

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

    # Print final aggregated results
    print("-"*80)
    print("AGGREGATED MOVE RANKINGS".center(80))
    print("-"*80 + "\n")
    
    for rank, (move, score) in enumerate(final_policy[:10], 1):
        bar_length = int(score * 50)
        bar = "█" * bar_length
        tera_indicator = " [TERA]" if "-tera" in move else ""
        print(f"{rank}. {move:<30} {score*100:>6.2f}% {bar}{tera_indicator}")
    
    if len(final_policy) > 10:
        print(f"... and {len(final_policy) - 10} more moves")
    
    # Handle Tera moves special logic
    top_move = final_policy[0][0]
    top_is_tera = "-tera" in top_move
    
    if top_is_tera:
        # Top move is Tera - only use it 70% of the time
        print("\n" + "!"*80)
        print("⚠️  TOP MOVE IS TERA - SPECIAL HANDLING".center(80))
        print("!"*80)
        
        # Find the highest non-tera move
        top_non_tera = next((move for move, score in final_policy if "-tera" not in move), None)
        
        if top_non_tera and random.random() < 0.30:
            # 30% chance: override to top non-tera move
            selected_move = top_non_tera
            print(f"30% override triggered: Selecting {top_non_tera} instead of {top_move}")
        else:
            # 70% chance: use the tera move
            selected_move = top_move
            print(f"70% case: Using Tera move {top_move}")
        
        print("!"*80 + "\n")
    else:
        # Top move is NOT Tera - filter out all Tera moves from the pool
        non_tera_moves = [(move, score) for move, score in final_policy if "-tera" not in move]
        
        if len(non_tera_moves) == 0:
            # Safety: if somehow all moves are Tera (shouldn't happen), use original
            non_tera_moves = final_policy
        
        print(f"\nFiltered out {len(final_policy) - len(non_tera_moves)} Tera moves from selection pool")
        
        # Extract moves and weights from non-tera pool
        moves = [move for move, score in non_tera_moves]
        weights = [score for move, score in non_tera_moves]
        
        # Draw a straw! Pure weighted random selection (no tera moves)
        selected_move = random.choices(moves, weights=weights, k=1)[0]
    
    # Find which rank the selected move was
    selected_rank = next(i for i, (move, _) in enumerate(final_policy, 1) if move == selected_move)
    selected_score = next(score for move, score in final_policy if move == selected_move)
    
    print("\n" + "="*80)
    print(f"RANDOMLY SELECTED: {selected_move} (Rank #{selected_rank}, {selected_score*100:.2f}%)".center(80))
    print("="*80 + "\n")

    logger.info("Top Choice in rankings:")
    logger.info(f"\t{round(final_policy[0][1] * 100, 3)}%: {final_policy[0][0]}")
    logger.info(f"Randomly selected: {selected_move} (rank #{selected_rank}, {round(selected_score * 100, 3)}%)")

    return selected_move


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

    default_num_battles_multiplier = 2
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

    print("\n" + "="*80)
    print("STARTING MCTS SEARCH".center(80))
    print("="*80)
    print(f"Battles: {num_battles} | Time per battle: {search_time_per_battle}ms | Time remaining: {battle.time_remaining}s")
    print("="*80 + "\n")
    
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