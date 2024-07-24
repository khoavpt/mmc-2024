import hydra
from omegaconf import DictConfig
import rootutils

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIGPATH = str(ROOTPATH / "configs")

import src.data.dataset as dataset
import src.algorithm.genetic_alg as ga
import src.algorithm.fitness_func as ff

@hydra.main(config_path=CONFIGPATH, config_name="config", version_base=None)
def main(cfg:DictConfig):
    # Load data
    data: dataset.Dataset = hydra.utils.instantiate(cfg.dataset)

    # Load fitness function
    _fitness_function: ff.FitnessFunction = hydra.utils.instantiate(cfg.fitness_function, dataset=data)
    def fitness_func(ga_instance, solution, solution_idx):
        return _fitness_function.fitness_function(solution)

    # Set initial population
    initial_population = ga.generate_feasible_population(cfg.genetic_algorithm.sol_per_pop, data.num_faculties, data.num_days, data.num_doctor1, data.num_doctor2, data.num_nurse, data.num_doctor1_per_day, data.num_doctor2_per_day, data.num_nurse_per_day)

    # Load algorithm
    num_genes = (data.num_doctor1 + data.num_doctor2 + data.num_nurse) * data.num_faculties * data.num_days
    algo: ga.GeneticAlgorithm = hydra.utils.instantiate(cfg.genetic_algorithm, 
                                                        fitness_func=fitness_func, 
                                                        num_genes=num_genes,
                                                        dataset=data)
                                                       
    algo.setup(initial_population)
    algo.run()
    algo.plot_results()


if __name__ == '__main__':
    main()