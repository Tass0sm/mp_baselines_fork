"""
OMPL Based Planner
"""
import abc

import numpy as np
import torch

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from mp_baselines.planners.base import MPPlanner
from mp_baselines.planners.utils import extend_path


INTERPOLATE_NUM = 64
DEFAULT_PLANNING_TIME = 5.0


class StateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class OMPLPlanner(MPPlanner):

    def __init__(
            self,
            name: str = "OMPLPlanner",
            planner_name: str = "RRTConnect",
            simplify_solution: bool = False,
            task=None,
            start_state_pos: torch.Tensor = None,
            goal_state_pos: torch.Tensor = None,
            allowed_time: float = DEFAULT_PLANNING_TIME,
            tensor_args: dict = None,
            **kwargs
    ):
        assert start_state_pos is not None and goal_state_pos is not None

        super(OMPLPlanner, self).__init__(name=name, tensor_args=tensor_args)
        self.task = task

        self.q_dim = task.robot.q_dim
        self.start_state_pos = start_state_pos
        self.goal_state_pos = goal_state_pos
        self.allowed_time = allowed_time
        self.simplify_solution = simplify_solution

        # OMPL Objects
        self.space = StateSpace(self.q_dim)

        min_q_bounds = task.robot.q_min.cpu().tolist()
        max_q_bounds = task.robot.q_max.cpu().tolist()
        bounds = ob.RealVectorBounds(self.q_dim)
        joint_bounds = zip(min_q_bounds, max_q_bounds)
        for i, (lower_limit, upper_limit) in enumerate(joint_bounds):
            bounds.setLow(i, lower_limit)
            bounds.setHigh(i, upper_limit)
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()

        if self.simplify_solution:
            self.ps = og.PathSimplifier(self.si)
        
        self.set_planner(planner_name)

    def is_state_valid(self, q):
        q_arr = torch.tensor([q[i] for i in range(self.q_dim)], **self.tensor_args)
        ret = self.task.compute_collision(q_arr).item()
        # if no collision, its valid
        return not bool(ret)

    def set_planner(self, planner_name):
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)


    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """
        return self._run_optimization(opt_iters, **observation)

    def _run_optimization(self, opt_iters, **observation):
        start = self.start_state_pos
        goal = self.goal_state_pos

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i].item()
            g[i] = goal[i].item()

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(self.allowed_time)
        res = False
        sol_path_list = []
        if solved:
            # print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()

            if self.simplify_solution:
                self.ps.simplify(sol_path_geometric, self.allowed_time)

            # sol_path_geometric.interpolate(INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            sol_path_arr = np.array(sol_path_list)
            print(sol_path_arr)
        else:
            return None

        return sol_path_arr

    def state_to_list(self, state):
        return [state[i] for i in range(self.q_dim)]

    def render(self, ax, **kwargs):
        # sol_path_geometric = self.ss.getSolutionPath()
        # sol_path_states = sol_path_geometric.getStates()
        # sol_path_list = [self.state_to_list(state) for state in sol_path_states]
        # for sol_path in sol_path_list:
        #     self.is_state_valid(sol_path)
        # sol_path_arr = np.array(sol_path_list)

        # for node in self.nodes_tree_1:
        #     node.render(ax)
        # for node in self.nodes_tree_2:
        #     node.render(ax)
        pass
