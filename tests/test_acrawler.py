import unittest
import numpy as np
from skimage.data import camera
from acrawler import ACEnvironment, ACAgent, ACSimulation


class ACEnvironmentTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_can_be_initialized_from_matrix(self):

        sample_arr = np.arange(25).reshape(5, 5)
        env = ACEnvironment(sample_arr)

        #Test values
        self.assertTrue(np.array_equal(sample_arr, env.values))

        #Test agent_models map
        expected = np.zeros((5, 5), dtype=bool)
        self.assertTrue(np.array_equiv(expected, env.population))


    def test_can_be_initialized_from_matrix_and_mask(self):
        sample_arr = np.arange(9).reshape(3, 3)
        sample_init_mask = np.array([
            [False, True, False],
            [False, False, False],
            [True, False, True],
        ])
        env = ACEnvironment(sample_arr, sample_init_mask)

        self.assertTrue(np.array_equal(sample_arr, env.values), "sample values differ from env values")
        self.assertTrue(np.array_equal(sample_init_mask, env.footprints),
                        "expected footprints differ from env footprints")

    def test_can_retrieve_crawler_map(self):
        sample_arr = np.arange(100).reshape(10, 10)
        sample_init_mask = np.random.randint(low=0, high=1, size=100).reshape(10, 10).astype(bool)

        env = ACEnvironment(sample_arr, sample_init_mask)

        self.assertTrue(np.array_equal(sample_init_mask, env.get_population_map()),
                        "Population map is not the expected")

    def test_can_return_population_as_a_list_of_references(self):
        arr = np.arange(25).reshape(5, 5)
        map = np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0]
        ]).astype(bool)

        env = ACEnvironment(arr, map)

        refs = env.get_population_references()

        x, y = np.where(map)
        self.assertEqual(len(refs), 9, "Size of list of references don't match!")
        pop_idx = range(len(refs))

        for k, i, j in zip(pop_idx, x, y):
            self.assertTrue(env.population[i, j] is refs[k],
                            "The reference in list do not point to the exact reference in environment")

    def test_can_return_energy_map(self):
        arr = np.arange(25).reshape(5, 5)
        map = np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0]
        ]).astype(bool)

        env = ACEnvironment(arr, map)
        energy_map = env.get_energy_map()
        self.assertTrue(np.array_equal(map * 5, energy_map))





class ACAgentTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_can_do__initialization(self):
        x = np.arange(25).reshape((5, 5))

        env = ACEnvironment(x)

        a = ACAgent((0, 0), env)
        expected_footprints = np.zeros((5, 5), dtype=bool)
        expected_footprints[0, 0] = True

        self.assertTrue(a is env.population[0, 0], "A is not the crawler in environment in A.position")
        self.assertTrue(np.array_equal(expected_footprints, env.footprints), "A didn't left a footprint")

    def test_cant_gain_more_energy_than_max(self):
        x = np.arange(25).reshape((5, 5))
        env = ACEnvironment(x)
        a = ACAgent((0, 0), env, max_energy=30, absorption_rate=10000000)

        a.update()

        self.assertEqual(a.energy_, 30, 'Agent have more energy than possible for it')





    def test_crawler_update_with_battle_win_and_kill(self):
        arr = np.arange(4).reshape(2, 2)
        map = np.array([
            [1, 0],
            [0, 1]
        ]).astype(bool)

        env = ACEnvironment(arr, map)

        ag1 = env.population[0, 0]  # The winner
        ag2 = env.population[1, 1]  # The looser

        ag1.update()

        self.assertTrue(env.population[1, 1] is ag1)
        self.assertEqual((1, 1), ag1.position_)
        self.assertTrue(ag2.energy_ == 0)

    def test_crawler_update_with_battle_lost_and_die(self):
        arr = np.arange(4).reshape(2, 2)
        map = np.array([
            [1, 0],
            [0, 1]
        ]).astype(bool)

        env = ACEnvironment(arr, map)

        ag1 = env.population[0, 0]  # The winner
        ag2 = env.population[1, 1]  # The looser
        ag2.energy_ = 10
        ag1.update()

        self.assertTrue(env.population[1, 1] is ag2)
        self.assertEqual((1, 1), ag2.position_)
        self.assertTrue(ag1.energy_ == 0)


    def test_can_battle_another_agent_and_win(self):
        arr = np.arange(4).reshape(2, 2)
        map = np.array([
            [1, 0],
            [0, 1]
        ]).astype(bool)

        env = ACEnvironment(arr, map)

        ag1 = env.population[0, 0]  # The winner
        ag2 = env.population[1, 1]  # The looser

        self.assertTrue(ag1.attack(ag2), "The agent lose the battle")
        self.assertEqual(ag2.energy_, 0, "The loser agent ag2 is alive!!!!KILL IT!!!!!!!!!!!!")

    def test_can_battle_another_agent_and_lost(self):
        arr = np.arange(4).reshape(2, 2)
        map = np.array([
            [1, 0],
            [0, 1]
        ]).astype(bool)

        env = ACEnvironment(arr, map)

        ag1 = env.population[0, 0]  # The winner
        ag2 = env.population[1, 1]  # The looser

        ag1.energy_ = 2
        ag2.energy_ = 3

        self.assertFalse(ag1.attack(ag2), "The agent won the battle")
        self.assertEqual(ag1.energy_, 0, "The loser agent ag1 is alive!!!!KILL IT!!!!!!!!!!!!")



    def test_can_do_perception(self):
        arr = np.array([
            [0, 1, 1, 2, 1],
            [2, 1, 4, 4, 1],
            [2, 3, 3, 5, 4],
            [2, 2, 3, 6, 4],
            [10, 9, 8, 7, 5]
        ])

        env = ACEnvironment(arr)

        a1 = ACAgent((0, 0), env)
        a2 = ACAgent((0, 2), env)
        a3 = ACAgent((4, 0), env)
        a4 = ACAgent((1, 4), env)

        self.assertTrue(
            np.array_equal(a1.perception(), [(1, 0)]),
            "A1 perception is wrong"
        )

        self.assertTrue(
            np.array_equal(a2.perception(), [(1, 2), (1, 3)]),
            "A2 perception is wrong"
        )

        self.assertTrue(
            np.array_equal(a3.perception(), []),
            "A3 perception is wrong"
        )
        k = a4.perception()
        self.assertTrue(
            np.array_equal(a4.perception(), [(2, 3)]),
            "A4 perception is wrong"
        )


    def test_can_get_neighborhood_with_boundary_check(self):
        arr = np.array([
            [0, 1, 1, 2, 1],
            [2, 1, 4, 4, 1],
            [2, 3, 3, 5, 4],
            [2, 2, 3, 6, 4],
            [10, 9, 8, 7, 5]
        ])

        env = ACEnvironment(arr)

        # Test over boundary
        a1 = ACAgent((0, 0), env)  # Top left
        a2 = ACAgent((4, 0), env)  # Bottom left
        a3 = ACAgent((4, 4), env)  # Bottom right
        a4 = ACAgent((0, 4), env)  # Top right
        a5 = ACAgent((0, 1), env)  # Top
        a6 = ACAgent((4, 3), env)  # Bottom
        a7 = ACAgent((2, 0), env)  # Left
        a8 = ACAgent((2, 4), env)  # Right
        a9 = ACAgent((1, 1), env)  # Interior

        self.assertTrue(
            np.array_equal(a1.get_neighborhood_positions(), [(0, 1), (1, 0), (1, 1)]),
            "A1 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a2.get_neighborhood_positions(), [(3, 0), (3, 1), (4, 1)]),
            "A2 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a3.get_neighborhood_positions(), [(3, 3), (3, 4), (4, 3)]),
            "A3 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a4.get_neighborhood_positions(), [(0, 3), (1, 3), (1, 4)]),
            "A4 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a5.get_neighborhood_positions(), [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]),
            "A5 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a6.get_neighborhood_positions(), [(3, 2), (3, 3), (3, 4), (4, 2), (4, 4)]),
            "A6 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a7.get_neighborhood_positions(), [(1, 0), (1, 1), (2, 1), (3, 0), (3, 1)]),
            "A7 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a8.get_neighborhood_positions(), [(1, 3), (1, 4), (2, 3), (3, 3), (3, 4)]),
            "A8 do not get the expected neighborhood positions")

        self.assertTrue(
            np.array_equal(a9.get_neighborhood_positions(),
                           [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]),
            "A9 do not get the expected neighborhood positions")

    def test_can_move_and_erase_reference_on_last_pos(self):
        arr = np.arange(4).reshape(2, 2)
        env = ACEnvironment(arr)
        crawler = ACAgent((0, 0), env)

        crawler.move_to((1, 1))

        self.assertFalse(env.population[0, 0] is crawler, "Crawler move and don't erase his reference in last position")

    def test_can_die(self):
        arr = np.arange(4).reshape(2, 2)

        env = ACEnvironment(arr)
        crawler = ACAgent((0, 0), env)

        crawler.die()

        self.assertFalse(env.population[0, 0], "Agent still have reference in environment!")
        self.assertEqual(crawler.energy_, 0, "Agent still have energy")

    def test_perception_do_not_return_self_position(self):
        arr = np.arange(4).reshape(2, 2)
        env = ACEnvironment(arr)
        arr[0, 0] = 3
        crawler = ACAgent((0, 0), env)

        l = crawler.perception()

        self.assertFalse(crawler.position_ in l, "Agent perception return own position")

    def test_neighborhood_return_self_position(self):
        arr = np.arange(4).reshape(2, 2)
        env = ACEnvironment(arr)
        arr[0, 0] = 3
        crawler = ACAgent((0, 0), env)
        l = crawler.get_neighborhood_positions()
        self.assertFalse(crawler.position_ in l, "Agent neighborhood return own position")

    def test_can_do_correct_moves(self):
        arr1 = np.array([
            [0, 1, 1, 2, 1],
            [2, 1, 4, 4, 5],
            [2, 3, 3, 5, 4],
            [2, 2, 3, 6, 4],
            [10, 9, 8, 7, 5]
        ])

        arr2 = np.array([
            [0, 2, 1, 2, 1],
            [2, 2, 4, 4, 5],
            [0, 3, 4, 5, 0],
            [2, 1, 3, 6, 6],
            [10, 9, 8, 8, 8]
        ])

        arr2_init = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0]
        ], dtype=bool)

        env1 = ACEnvironment(arr1)
        env2 = ACEnvironment(arr2, arr2_init)

        a1 = ACAgent((0, 0), env1, absorption_rate=1)
        a2 = ACAgent((1, 4), env1, absorption_rate=1)
        a3 = ACAgent((0, 0), env2, absorption_rate=1)
        a4 = ACAgent((1, 4), env2, absorption_rate=1)

        expected_a1_positions = [(1, 0), (2, 1), (1, 2), (2, 3), (3, 3), (4, 2), (4, 1), (4, 0)]
        expected_a2_positions = [(1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)]
        expected_a3_positions = [(1, 0), (2, 1), (1, 2), (2, 3), (3, 3), (4, 2), (4, 1), (4, 0)]
        expected_a4_positions = [(1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)]

        a1_positions = []
        a2_positions = []
        a3_positions = []
        a4_positions = []

        for i in range(8):
            a1.update()
            a2.update()
            a3.update()
            a4.update()
            a1_positions.append(a1.position_)
            a2_positions.append(a2.position_)
            a3_positions.append(a3.position_)
            a4_positions.append(a4.position_)

        self.assertTrue(np.array_equal(a1_positions, expected_a1_positions), "A1 did wrong moves")
        self.assertTrue(np.array_equal(a2_positions, expected_a2_positions), "A2 did wrong moves")
        self.assertTrue(np.array_equal(a3_positions, expected_a3_positions), "A3 did wrong moves")
        self.assertTrue(np.array_equal(a4_positions, expected_a4_positions), "A4 did wrong moves")

    def test_lost_energy_after_move(self):
        arr = np.arange(4).reshape(2, 2)

        env = ACEnvironment(arr)
        crawler = ACAgent((0, 0), env)

        energy = crawler.energy_
        crawler.move_to((1, 1))

        self.assertEqual(crawler.energy_, energy - 1)

    def test_can_do_update_with_energy_absorption(self):
        arr = np.arange(4).reshape(2, 2)

        env = ACEnvironment(arr)
        crawler = ACAgent((0, 0), env, absorption_rate=1)

        energy = crawler.energy_
        crawler.update()

        absorption_comp = env.values[crawler.position_] * crawler.absorption_rate_
        self.assertEqual(crawler.energy_, energy + absorption_comp - 1)


class ACSimulationTest(unittest.TestCase):
    def setUp(self):
        self.test_environment = camera()[:50, :50]

    def tearDown(self):
        pass

    def test_can_run_simulation_100x(self):
        simulation = ACSimulation(environment=self.test_environment,
                                  initialization_map=np.ones(self.test_environment.shape).astype(bool),
                                  initial_energy=5,
                                  max_energy=5,
                                  absorption_rate=0.01,
                                  iterations=100)

        # Execute the simulation
        simulation.run()

        self.assertEqual(simulation.iterations_, 100)

    def test_can_run_until_100x_or_equilibrium(self):
        simulation = ACSimulation(environment=self.test_environment,
                                  initialization_map=np.ones(self.test_environment.shape).astype(bool),
                                  initial_energy=5,
                                  max_energy=5,
                                  absorption_rate=0.01,
                                  iterations=100,
                                  stop_condition='equilibrium')

        # Execute the simulation
        simulation.run()

        arr = simulation.environment_.get_population_map().astype(int)

        self.assertLessEqual(simulation.iterations_, 100)
        if simulation.iterations_ < 100:
            self.assertTrue(simulation.equilibrium_)

    def test_can_do_agent_initialization_by_map(self):
        n_crawlers = self.test_environment.size

        x = np.random.randint(low=0, high=self.test_environment.shape[0] - 1, size=n_crawlers)
        y = np.random.randint(low=0, high=self.test_environment.shape[1] - 1, size=n_crawlers)

        map = np.zeros(self.test_environment.shape).astype(bool)
        map[x, y] = True

        simulation = ACSimulation(environment=self.test_environment,
                                  initialization_map=map,
                                  initial_energy=5,
                                  max_energy=5,
                                  absorption_rate=0.01,
                                  iterations=100,
                                  stop_condition='equilibrium')

        self.assertTrue(np.array_equal(simulation.environment_.get_population_map(), map))


    def test_can_save_crawler_state_to_map(self):

        n_crawlers = self.test_environment.size

        x = np.random.randint(low=0, high=self.test_environment.shape[0] - 1, size=n_crawlers)
        y = np.random.randint(low=0, high=self.test_environment.shape[1] - 1, size=n_crawlers)

        map = np.zeros(self.test_environment.shape).astype(bool)
        map[x, y] = True

        simulation = ACSimulation(environment=self.test_environment,
                                  initialization_map=map,
                                  initial_energy=5,
                                  max_energy=5,
                                  absorption_rate=0.01,
                                  iterations=100,
                                  stop_condition='equilibrium')

        simulation.run()

