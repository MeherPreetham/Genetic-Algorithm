#!/usr/bin/env python3
import os, random, time, copy
from statistics import mean, pstdev
import matplotlib.pyplot as plt
from deap import base, creator, tools
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tasks",     type=int,   default=int(os.getenv("NUM_TASKS",1000)),  help="Number of tasks to schedule")
    p.add_argument("--num-cores",     type=int,   default=int(os.getenv("NUM_CORES",16)),
                   help="Number of identical cores")
    p.add_argument("--num-population", type=int,   default=int(os.getenv("NUM_POPULATION",200)),
                   help="GA population size")
    p.add_argument("--num-generations",   type=int,   default=int(os.getenv("NUM_GENERATIONS",500)),
                   help="Max GA generations")
    p.add_argument("--crossover-rate",       type=float, default=float(os.getenv("CROSSOVER_RATE",0.8)),
                   help="Crossover probability")
    p.add_argument("--mutation-rate",      type=float, default=float(os.getenv("MUTATION_RATE",0.2)),
                   help="Mutation probability")
    p.add_argument("--base-energy",   type=float, default=float(os.getenv("BASE_ENERGY",0.01)),
                   help="Joules per ms when core is active")
    p.add_argument("--idle-energy",    type=float, default=float(os.getenv("IDLE_ENERGY",0.002)),
                   help="Joules per ms when core is idle")
    p.add_argument("--stagnation-limit",    type=int,   default=int(os.getenv("STAGNATION_LIMIT",50)),
                   help="Early-stop if no improvement for N generations")
    p.add_argument("--seed",          type=int,   default=int(os.getenv("SEED",42)),
                   help="Random seed for reproducibility")
    return p.parse_args()

def setup_deap():
  try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMin)
  except RuntimeError:
    pass

def make_toolbox(args, execution_times):
  
  toolbox = base.Toolbox()
  toolbox.register("attr_core", random.randint, 0, args.num_cores-1)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_core, n = args.num_tasks)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = args.num_population)

  TOTAL_EXEC    = sum(execution_times)
  MEAN_LOAD     = TOTAL_EXEC / args.num_cores if args.num_cores > 0 else 1e-9
  MAX_MAKESPAN  = TOTAL_EXEC
  MAX_ENERGY = (
  TOTAL_EXEC * args.base_energy
    + (args.num_cores - 1) * TOTAL_EXEC * args.idle_energy
)

  # Weights.
  W_MAKESPAN = 0.4
  W_ENERGY = 0.2
  W_IMBALANCE = 0.4

  def evaluate(individual):
  
      # 1) Per-core total time & energy.
      core_times  = [0.0]*args.num_cores
      core_energy = [0.0]*args.num_cores

      for t_idx, core in enumerate(individual):
          t = execution_times[t_idx]
          core_times[core]  += t
          core_energy[core] += t * args.base_energy

      makespan = max(core_times) # worst-loaded core time.

      active_energy = sum(ct* args.base_energy for ct in core_times)
      idle_energy = sum((makespan - ct) * args.idle_energy for ct in core_times)
      total_energy = active_energy + idle_energy

      # 2 Imbalance via coefficient of variation.
      imbalance = pstdev(core_times) / MEAN_LOAD if TOTAL_EXEC > 0 else 0.0

      # 3 Normalized objectives
      nm = min(makespan / MAX_MAKESPAN, 1.0)
      ne = min(total_energy / MAX_ENERGY, 1.0)
      ni = min(imbalance, 1.0)

      # 4 Combined fitness (we want to MINIMISE this).
      score = (W_MAKESPAN*nm + W_ENERGY*ne + W_IMBALANCE*ni)

      # 5 Diagnostics on the individual.
      individual.core_times     = core_times
      individual.total_energy   = total_energy
      individual.makespan       = makespan
      individual.imbalance      = imbalance
      individual.fitness_value  = score

      return (score,)
  
  # Registering it.
  toolbox.register("evaluate", evaluate)
  # 3 Genetic Operators (crossover, mutation, and selection).
  toolbox.register("mate", tools.cxUniform, indpb = args.crossover_rate)
  toolbox.register("mutate", tools.mutUniformInt, low=0, up=args.num_cores-1, indpb= args.mutation_rate)
  toolbox.register("select", tools.selTournament, tournsize=3)
  toolbox.register("clone", copy.deepcopy)

  return toolbox

def main(args, toolbox):
  population = toolbox.population()
  for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)

  hall = tools.HallOfFame(1)
  best_per_gen = []
  avg_per_gen = []
  std_per_gen   = []
  stagnation    = 0
  best_so_far   = max(ind.fitness.values[0] for ind in population)

  start_time = time.time()

  LOG_INTERVAL = 10 # Printing the best generations every 10 generations.
  for gen in range(1, args.num_generations + 1):
    hall.update(population)
    current_best = hall[0].fitness.values[0]
    if current_best < best_so_far:
      best_so_far = current_best
      stagnation = 0
    else:
      stagnation += 1

    # Early stop if no improvement.
    if stagnation >= args.stagnation_limit:
      print(f"Early stopping at generation {gen}.")
      break

    # 3.1 select & clone.
    offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

    # 3.2 Crossover.
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < args.crossover_rate:
        toolbox.mate(child1, child2)
        del child1.fitness.values, child2.fitness.values

    # 3.3 Mutation.
    for mutant in offspring:
      if random.random() < args.mutation_rate:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    # 3.4 evaluating any individuals with invalid fitness.
    invalids = [ch for ch in offspring if not ch.fitness.valid]
    for ch in invalids:
      ch.fitness.values = toolbox.evaluate(ch)

    # Elitism
    offspring[0] = hall[0]

    # 3.5 replacing old population.
    population[:] = offspring

    fits = [ind.fitness.values[0] for ind in population]
    best_per_gen.append(max(fits))
    avg_per_gen.append(mean(fits))
    std_per_gen.append(pstdev(fits))

    # 3.6 Logging.
    if gen % LOG_INTERVAL == 0 or gen == args.num_generations:
      print(f"Gen {gen:4d} → Best={best_per_gen[-1]:.3f} | "
                f"Avg={avg_per_gen[-1]:.3f} | Std={std_per_gen[-1]:.3f}")

  elapsed = time.time() - start_time

  # 3.7 Final best solution.
  best = hall[0]
  print("\n=== FINAL BEST SOLUTION ===")
  print(f"Total time       : {elapsed:.2f}s")
  print(f"Best fitness     : {best.fitness.values[0]:.3f}")
  print(f"Chromosome[:10]  : {best[:10]} …")
  print(f"Makespan         : {best.makespan:.2f} ms")
  print(f"Total energy (J) : {best.total_energy:.3f}")
  print(f"Imbalance        : {best.imbalance:.3f}")

  plt.plot(best_per_gen, label="Best")
  plt.plot(avg_per_gen,  label="Average")
  plt.xlabel("Generation"); plt.ylabel("Fitness")
  plt.legend()
  plt.show()
  plt.tight_layout()

if __name__ == "__main__":
  args = parse_args()
  random.seed(args.seed)
  execution_times = [random.randint(1, 10) for _ in range(args.num_tasks)]
  setup_deap()
  toolbox = make_toolbox(args, execution_times)
  main(args, toolbox)
  