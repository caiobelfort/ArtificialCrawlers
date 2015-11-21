__author__ = 'caiob'

import numpy as np


class ACAgent:
    def __init__(self, position, environment, energy=5, max_energy=100, absorption_rate=0.1, move_type='peak'):
        """
        Create a crawler
        :param position: tuple (x, y)
        :param environment: ACEnvironment
        """
        self.position_ = position
        self.environment_ = environment
        self.energy_ = energy
        self.absorption_rate_ = absorption_rate
        self.max_energy_ = max_energy
        self.environment_.population[self.position_] = self
        self.environment_.footprints[self.position_] = True
        self.move_type = move_type

    def get_neighborhood_positions(self):
        """
        Returns a list of positions which the crawler can move
        """
        l = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                l.append((self.position_[0] + i, self.position_[1] + j))

        # Eliminate the out of boundary positions
        lb = []
        env_shape = self.environment_.values.shape
        for pos in l:
            x_off = pos[0] < 0 or pos[0] >= env_shape[0]
            y_off = pos[1] < 0 or pos[1] >= env_shape[1]

            if not (x_off or y_off):
                lb.append(pos)
        return lb

    def perception(self):
        """
        The crawler perceives his neighbor environment and look for best cell to move.
        :return:  True if the crawler moves, false otherwise
        """
        neighborhood_positions = self.get_neighborhood_positions()

        auxiliary_array = []    # Used to keep the values in the neighborhood

        # Get mapped values
        for pos in neighborhood_positions:
            auxiliary_array.append(self.environment_.values[pos])
        auxiliary_array = np.array(auxiliary_array)

        best_value = None
        if self.move_type == 'peak':
            best_value = auxiliary_array.max()  # Get the higher value in the neighborhood
        elif self.move_type == 'valley':
            best_value = auxiliary_array.min()  # Get the lower value in the neighborhood

        best_positions = []     # Used to keep the best position(s)

        # If the actual position is better the all neighborhood, the crawler do nothing.
        # We only include positions which the value (fitness) is better than the actual.
        if best_value > self.environment_.values[self.position_]:
            bx, = np.where(auxiliary_array == best_value)  # Get the position where the value is equal the best
            for i in bx:
                best_positions.append(neighborhood_positions[i])

        return best_positions

    def update(self):
        move_choices = self.perception()
        n_choices = len(move_choices)

        # We only update the crawler if we have a choice of movement, otherwise the crawler will stay in original place.
        if n_choices >= 1:
            follow_moves = []       # Choices where the crawler follow the steps of another crawler.
            battle_moves = []       # Choices where the crawler need fight against other crawler for the new position.
            explorer_moves = []     # Choices where the crawler go to a position where no one was.

            # A crawler always prefers move to a free cell, so we separate the free cells from battle cells
            for choice in move_choices:
                # Has other agent?
                if self.environment_.population[choice]:
                    battle_moves.append(choice)  # Is a battle move
                # No, other agent was here in the past?
                elif self.environment_.footprints[choice]:
                    follow_moves.append(choice)  # Is a follow move
                # The agent is the first to reach this cell, it is a explorer! :)
                else:
                    explorer_moves.append(choice)

            # Always try to move to a place where other crawler was in the pas, and this place is free.
            if follow_moves:
                self.move_to(follow_moves[0])
                self.absorb()

            # If exist a place where no other crawler was in the past, move to it
            elif explorer_moves:
                # Here is battle, for now just move
                self.move_to(explorer_moves[0])
                self.absorb()

            # There's only one choice, fight against other crawler, so attack that bastard!!!
            else:
                enemy = self.environment_.population[battle_moves[0]]
                if self.attack(enemy):  # Attack, if win, move
                    self.move_to(battle_moves[0])
                    self.absorb()

            return True
        return False

    def move_to(self, pos):

        if pos == self.position_:
            return False

        last_pos = self.position_
        self.position_ = pos
        self.environment_.population[last_pos] = False
        self.environment_.population[self.position_] = self
        self.energy_ -= 1

        return True

    def absorb(self):
        self.energy_ += self.absorption_rate_ * self.environment_.values[self.position_]
        if self.energy_ > self.max_energy_:
            self.energy_ = self.max_energy_

    def die(self):
        self.environment_.population[self.position_] = False

    def attack(self, ag):
        if self.energy_ >= ag.energy_:
            ag.die()
            return True
        self.die()
        return False

    def to_array(self):
        """
        Transforms the crawler object in a numpy array with the x, y , energy values
        :return: ndarray
        """
        return np.array([self.position_[0], self.position_[1], self.energy_])


class ACEnvironment:
    """
    This class represents the crawlers model ambient
    """

    def __init__(self, values, population_mask=None, params=[]):
        """
        If initialization_map is None, no crawlers will be initialized in the environment, but the map will
        be created
        :param values: ndarray of floats
        :param initialization_map: ndarray or None
        """

        self.values = values

        # First create a ndarray of zeros
        self.population = np.zeros(self.values.shape, dtype=object)
        self.footprints = np.zeros(self.values.shape, dtype=bool)

        # If a initialize mask was passed, initialize the population
        if not population_mask is None:
            x, y = np.where(population_mask)
            for position in zip(x, y):
                if len(params) == 3:
                    ACAgent(position, self, energy=params[0], max_energy=params[1], absorption_rate=params[2])
                else:
                    ACAgent(position, self)

    def shape(self):
        return self.values.shape

    def get_population_map(self):
        map_ = np.zeros(self.population.shape, dtype=bool)
        x, y = self.population.shape
        for i in range(x):
            for j in range(y):
                if self.population[i, j]:
                    map_[i, j] = True
        return map_

    def get_population_references(self):
        refs = []
        r, c, = self.population.shape

        # Search for cells with agents
        for i in range(r):
            for j in range(c):
                if self.population[i, j]:
                    refs.append(self.population[i, j])

        return refs

    def get_energy_map(self):
        r, c, = self.population.shape
        energy_map = np.zeros(self.population.shape).astype(int)
        for i in range(r):
            for j in range(c):
                if self.population[i, j]:
                    energy_map[i, j] = self.population[i, j].energy_

        return energy_map


class ACSimulation:
    def __init__(self,
                 environment,
                 initialization_map,
                 initial_energy=5,
                 max_energy=100,
                 absorption_rate=0.1,
                 iterations=1000,
                 stop_condition=''):

        self.environment_ = ACEnvironment(environment, population_mask=initialization_map, params=[initial_energy,
                                                                                                   max_energy,
                                                                                                   absorption_rate])
        self.population_ = self.environment_.get_population_references()
        self.graveyard = []  # Keeps the position and energy of dead crawlers

        self.max_iterations_ = iterations
        self.iterations_ = 0
        self.stop_condition_ = stop_condition
        self.equilibrium_ = False
        self.observers = []

    def run(self):

        while self.iterations_ < self.max_iterations_:

            if self.stop_condition_ == 'equilibrium':
                if self.equilibrium_:
                    return

            self.update()
            self.iterations_ += 1

    def register_observer(self, observer):
        """
        Add a new observer to the simulation
        :param observer: An object who receives simulation updates
        :return: None
        """
        self.observers.append(observer)

    def notify_all(self):
        """
        Notify all observer about the current state
        :return:
        """
        current_state = self.get_state()
        for observer in self.observers:
            observer.notify(current_state)

    def get_state(self):
        """
        Get the current state of the simulation
        :return:
        """
        live_crawlers = []
        dead_crawlers = []

        for crawler in self.population_:
            live_crawlers.append(crawler.to_array())

        for crawler in self.graveyard:
            dead_crawlers.append(crawler.to_array())

        state = {'iter': self.iterations_, 'equilibrium': self.equilibrium_, 'live': live_crawlers, 'dead': dead_crawlers}

        return state

    def update(self):
        flag = False

        sz = len(self.population_)
        k = 0
        while k < sz:
            if self.population_[k].update():
                flag = True

            if self.population_[k].energy_ == 0 or \
                    not (self.environment_.population[self.population_[k].position_] is self.population_[k]):
                # Without energy or lost his reference in environment the crawler is dead
                self.graveyard.append(self.population_[k])
                del self.population_[k]
                sz -= 1
            else:
                k += 1

        self.equilibrium_ = not flag
        self.notify_all()


class Observer:
    """
    Base class for simulation observers
    """
    def notify(self, state):
        raise NotImplementedError


class FinalStateObserver(Observer):
    """
    Save the final state of the simulation
    """

    def notify(self, state):
        if state['equilibrium']:
            live = np.ones(state['live'].shape[0], dtype=int)
            dead = np.ones(state['dead'].shape[0], dtype=int)

            live_or_dead = np.concatenate((live, dead))
            crawlers = np.vstack((state['live'], state['dead']))
            result = np.hstack((crawlers, live_or_dead))

