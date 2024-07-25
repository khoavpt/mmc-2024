import hydra
from omegaconf import DictConfig
import rootutils
import optuna
import logging

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIGPATH = str(ROOTPATH / "configs")

import src.data.dataset as dataset
import src.algorithm.genetic_alg as ga
import src.algorithm.fitness_func as ff
import src.algorithm.crossover_func as cf
import src.algorithm.mutation_func as mf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, cfg):
    sol_per_pop = trial.suggest_categorical('sol_per_pop', [20, 50, 70, 100])
    num_parents_mating = trial.suggest_categorical('num_parents_mating', [5, 10, 15, 20])
    mutation_percent_genes = trial.suggest_categorical('mutation_percent_genes', [5, 10, 20, 30])
    parent_selection_type = trial.suggest_categorical('parent_selection_type', ['sss', 'rws', 'tournament'])

    # Load data
    data: dataset.Dataset = hydra.utils.instantiate(cfg.dataset)

    # Load fitness function
    fitness_func: ff.CustomFitnessFunction = hydra.utils.instantiate(cfg.fitness_function, dataset=data)

    # Load crossover and mutation functions
    crossover_func = cf.CustomCrossoverFunc(data)
    mutation_func = mf.CustomMutationFunc(data)

    # Set initial population
    initial_population = mf.generate_feasible_population(data, sol_per_pop)

    # Load algorithm with the trial's hyperparameters
    num_genes = (data.num_doctor1 + data.num_doctor2 + data.num_nurse) * data.num_faculties * data.num_days
    algo: ga.GeneticAlgorithm = ga.GeneticAlgorithm(num_genes=num_genes,
                                                     fitness_func=fitness_func,
                                                     crossover_func=crossover_func,
                                                     mutation_func=mutation_func,
                                                     dataset=data,
                                                     num_generations=cfg.genetic_algorithm.num_generations,
                                                     num_parents_mating=num_parents_mating,
                                                     sol_per_pop=sol_per_pop,
                                                     parent_selection_type=parent_selection_type,
                                                     mutation_percent_genes=mutation_percent_genes)
    algo.setup(initial_population)
    algo.run()

    best_fitness = max(algo.fitness_values)
    print(f"Best fitness: {best_fitness}")
    return best_fitness

@hydra.main(config_path=CONFIGPATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cfg), n_trials=50)

    # Log the best parameters
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best params: {study.best_trial.params}")
    logger.info(f"Best fitness: {study.best_trial.value}")

if __name__ == '__main__':
    main()