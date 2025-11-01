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


def select_move_from_mcts_results(mcts_results: list, battle=None) -> str:
    final_policy = {}
    
    # Print header
    print("\n" + "="*80)
    print("MCTS SEARCH RESULTS".center(80))
    print("="*80 + "\n")
    
    # Track which moves won each policy
    policy_winners = []
    
    # Print individual policy results
    for mcts_result, sample_chance, index in mcts_results:
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        visit_pct = round(100 * this_policy.visits / mcts_result.total_visits, 2)
        avg_score = round(this_policy.total_score / this_policy.visits, 3)
        sample_mult = round(sample_chance, 3)
        
        policy_winners.append(this_policy.move_choice)
        
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
    
    for rank, (move, score) in enumerate(final_policy[:5], 1):
        bar_length = int(score * 50)
        bar = "█" * bar_length
        print(f"{rank}. {move:<30} {score*100:>6.2f}% {bar}")
    
    # Calculate confidence metrics
    top_move, top_score = final_policy[0]
    second_score = final_policy[1][1] if len(final_policy) > 1 else 0
    score_gap = top_score - second_score
    
    # Count how many policies agreed on the top move
    num_policies = len(policy_winners)
    agreement_count = policy_winners.count(top_move)
    agreement_pct = agreement_count / num_policies
    
    print("\n" + "-"*80)
    print("CONFIDENCE ANALYSIS".center(80))
    print("-"*80)
    print(f"Top move score: {top_score*100:.2f}%")
    print(f"Score gap to 2nd: {score_gap*100:.2f}%")
    print(f"Policy agreement: {agreement_count}/{num_policies} ({agreement_pct*100:.1f}%)")
    
    # Determine if we should be deterministic or random
    if top_score >= 0.70 and score_gap >= 0.30:
        # VERY HIGH CONFIDENCE - Always pick it
        confidence = "VERY HIGH"
        temperature = 0.0
        reason = "Dominant move across all scenarios"
    elif top_score >= 0.65 and score_gap >= 0.25:
        # HIGH CONFIDENCE - Mostly pick it
        confidence = "HIGH"
        temperature = 0.05
        reason = "Strong consensus with clear advantage"
    elif top_score >= 0.55 and score_gap >= 0.20:
        # GOOD CONFIDENCE - Usually pick it
        confidence = "GOOD"
        temperature = 0.10
        reason = "Clear favorite with meaningful gap"
    elif top_score >= 0.50 and score_gap >= 0.15:
        # MODERATE CONFIDENCE - Often pick it
        confidence = "MODERATE"
        temperature = 0.15
        reason = "Preferred move but alternatives viable"
    elif score_gap >= 0.10:
        # LOW-MODERATE CONFIDENCE
        confidence = "LOW-MODERATE"
        temperature = 0.20
        reason = "Slight preference, multiple good options"
    else:
        # LOW CONFIDENCE - Randomize heavily
        confidence = "LOW"
        temperature = 0.25
        reason = "Close decision, no clear best move"
    
    print(f"Confidence level: {confidence}")
    print(f"Reason: {reason}")
    print(f"Temperature: {temperature}")
    
    # Select move based on confidence
    if temperature == 0.0:
        selected_move = top_move
        print(f"Decision: DETERMINISTIC (always pick top move)")
    else:
        # Consider top moves within reasonable range
        threshold = max(top_score * 0.6, second_score)  # At least 60% of top or better than 2nd
        candidates = [(move, score) for move, score in final_policy 
                      if score >= threshold]
        
        if len(candidates) == 1 or temperature < 0.08:
            # Only one viable candidate or very low temperature
            selected_move = top_move
            print(f"Decision: DETERMINISTIC (temperature too low for randomization)")
        else:
            # Apply temperature and sample
            scores = [score ** (1/temperature) for _, score in candidates]
            total = sum(scores)
            probs = [s/total for s in scores]
            
            choice_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            selected_move = candidates[choice_idx][0]
            
            print(f"Decision: STOCHASTIC")
            print(f"Candidates considered: {len(candidates)}")
            print(f"Probabilities:")
            for i, ((move, orig_score), prob) in enumerate(zip(candidates[:3], probs[:3])):
                print(f"  {move}: {prob*100:.1f}% chance (original: {orig_score*100:.1f}%)")
    
    print("\n" + "="*80)
    print(f"SELECTED MOVE: {selected_move}".center(80))
    print("="*80 + "\n")

    logger.info("Top Choice:")
    logger.info(f"\t{round(final_policy[0][1] * 100, 3)}%: {final_policy[0][0]}")
    logger.info(f"Confidence: {confidence}, Temperature: {temperature}")
    logger.info(f"Selected: {selected_move}")

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