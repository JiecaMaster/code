"""
This module defines various agent classes for a game, including random agents, greedy agents.
You need to implement your own agent in the YourAgent class using minimax algorithms.

Classes:
    Agent: Base class for all agents.
    RandomAgent: Agent that selects actions randomly.
    SimpleGreedyAgent: Greedy agent that selects actions based on maximum vertical advance.
    YourAgent: Placeholder for user-defined agent.

Class Agent:
    Methods:
        __init__(self, game): Initializes the agent with the game instance.
        getAction(self, state): Abstract method to get the action for the current state.
        oppAction(self, state): Abstract method to get the opponent's action for the current state.

Class RandomAgent(Agent):
    Methods:
        getAction(self, state): Selects a random legal action.
        oppAction(self, state): Selects a random legal action for the opponent.

Class SimpleGreedyAgent(Agent):
    Methods:
        getAction(self, state): Selects an action with the maximum vertical advance.
        oppAction(self, state): Selects an action with the minimum vertical advance for the opponent.

Class YourAgent(Agent):
    Methods:
        getAction(self, state): Placeholder for user-defined action selection.
        oppAction(self, state): Placeholder for user-defined opponent action selection.
"""

import random, re, datetime
import board


class Agent(object):
    def __init__(self, game):
        self.game = game
        self.action = None

    def getAction(self, state):
        raise Exception("Not implemented yet")

    def oppAction(self, state):
        raise Exception("Not implemented yet")


class RandomAgent(Agent):

    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

    def oppAction(self, state):
        legal_actions = self.game.actions(state)
        self.opp_action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
    # a one-step-lookahead greedy agent that returns action with max vertical advance

    def getAction(self, state):

        legal_actions = self.game.actions(state)

        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)

    def oppAction(self, state):
        legal_actions = self.game.actions(state)

        self.opp_action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            min_vertical_advance_one_step = min([action[0][0] - action[1][0] for action in legal_actions])
            min_actions = [action for action in legal_actions if action[0][0] - action[1][0] == min_vertical_advance_one_step]
        else:
            min_vertical_advance_one_step = min([action[1][0] - action[0][0] for action in legal_actions])
            min_actions = [action for action in legal_actions if action[1][0] - action[0][0] == min_vertical_advance_one_step]

        self.opp_action = random.choice(min_actions)

class YourAgent(Agent):
    dep = 2
    range = 0.1
    def getAction(self, state):
        player = self.game.player(state)
        depth = self.dep  # 深度限制为2
        alpha = float('-inf')
        beta = float('inf')
        self.action = None

        # 获取所有可能的动作及其评估值
        action_evals = []
        for action in self.game.actions(state):
            next_state = self.game.succ(state, action)
            eval = self.minimax(next_state, depth - 1, alpha, beta, False, player, [action])
            action_evals.append((action, eval))

        if not action_evals:
            # 如果没有合法动作，随机选择一个动作
            legal_actions = self.game.actions(state)
            self.action = random.choice(legal_actions)
            return

        # 找到最高的评估值
        max_eval = max(eval for _, eval in action_evals)

        # 收集所有评估值在 max_eval - 2.5 以上的动作
        threshold = max_eval - self.range
        top_actions = [action for action, eval in action_evals if eval >= threshold]

        # 随机选择一个动作
        self.action = random.choice(top_actions)

    def oppAction(self, state):
        player = self.game.player(state)
        depth = self.dep  # 深度限制为2
        alpha = float('-inf')
        beta = float('inf')
        self.opp_action = None

        # 获取所有可能的对手动作及其评估值
        action_evals = []
        for action in self.game.opp_actions(state):
            next_state = self.game.opp_succ(state, action, last_action=action[1])
            eval = self.minimax(next_state, depth - 1, alpha, beta, True, 3 - player, [action])
            action_evals.append((action, eval))

        if not action_evals:
            # 如果没有合法动作，随机选择一个动作
            legal_actions = self.game.opp_actions(state)
            self.opp_action = random.choice(legal_actions)
            return

        # 找到最低的评估值
        min_eval = min(evaluate for _, evaluate in action_evals)

        # 收集所有评估值在 min_eval + 2.5 以下的动作
        threshold = min_eval + self.range
        top_actions = [action for action, evaluate in action_evals if evaluate <= threshold]

        # 随机选择一个动作
        self.opp_action = random.choice(top_actions)

    def minimax(self, state, depth, alpha, beta, maximizingPlayer, player, previous_positions):
        if depth == 0:#or self.game.isEnd(state, 0):
            return self.evaluation(state, player, previous_positions)

        legal_actions = self.game.actions(state) if maximizingPlayer else self.game.opp_actions(state)

        if not legal_actions:
            return self.evaluation(state, player, previous_positions)

        if maximizingPlayer:
            maxEval = float('-inf')
            for action in legal_actions:
                if action in previous_positions:
                    continue
                next_state = self.game.succ(state, action)
                evaluate = self.minimax(next_state, depth - 1, alpha, beta, False, player,
                                        previous_positions + [action])
                maxEval = max(maxEval, evaluate)
                alpha = max(alpha, evaluate)
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = float('inf')
            for action in legal_actions:
                if action in previous_positions:
                    continue
                next_state = self.game.opp_succ(state, action, last_action=action[1])
                evaluate = self.minimax(next_state, depth - 1, alpha, beta, True, player, previous_positions + [action])
                minEval = min(minEval, evaluate)
                beta = min(beta, evaluate)
                if beta <= alpha:
                    break
            return minEval

    def evaluation(self, state, player, previous_positions):
        inboard = state[1]
        opponent = 3 - player
        player_positions = inboard.getPlayerPiecePositions(player)
        opponent_positions = inboard.getPlayerPiecePositions(opponent)

        # 计算我方棋子到目标区域的总距离
        player_score = 0
        for pos in player_positions:
            player_score += self.distance_to_goal(pos, player, inboard)

        # 计算对方棋子到目标区域的总距离
        opponent_score = 0
        for pos in opponent_positions:
            opponent_score += self.distance_to_goal(pos, opponent, inboard)

        # 后退行为的惩罚
        backward_penalty = 0
        for action in previous_positions:
            if self.is_backward_move(action, player):
                backward_penalty += 5  # 增加惩罚值至15

        #普通棋子占据特殊棋子位置的惩罚
        special_penalty = 0
        # 定义对手的特殊位置
        size = self.game.board.size  # 获取棋盘大小
        if opponent == 2:
            opponent_special_positions = [(2, 1), (2, 2)]
            player_special_positions = [(2 * size - 2, 1), (2 * size - 2, 2)]
        else:
            opponent_special_positions = [(2 * size - 2, 1), (2 * size - 2, 2)]
            player_special_positions = [(2, 1), (2, 2)]

        for pos in player_positions:
            # 检查该位置是否为我方普通棋子
            if inboard.board_status[pos] == player:
                # 检查该普通棋子是否占据了对手的特殊位置
                if pos in opponent_special_positions:
                    special_penalty += 50  # 增加较大的惩罚值
        for pos in opponent_positions:
            if inboard.board_status[pos] == opponent + 2:
                if pos in player_special_positions:
                    special_penalty += 50
        # 特殊棋子占据棋盘端点位置的惩罚
        endpoint_penalty = 0
        # 定义棋盘端点位置
        endpoint_positions_p1 = [(1, 1)]  # player1 的特殊棋子不允许占据的位置
        endpoint_positions_p2 = [(2 * size - 1, 1)]  # player2 的特殊棋子不允许占据的位置

        for pos in player_positions:
            # 检查该位置是否为我方特殊棋子
            if inboard.board_status[pos] == player + 2:  # player3 是 player1 的特殊棋子
                if player == 1:
                    if pos in endpoint_positions_p1:
                        endpoint_penalty += 50  # 增加较大的惩罚值
                elif player == 2:  # player4 是 player2 的特殊棋子
                    if pos in endpoint_positions_p2:
                        endpoint_penalty += 10  # 增加较大的惩罚值
        # 特殊棋子加分
        special_piece_bonus = 0
        for pos in player_positions:
            if inboard.board_status[pos] == player + 2:
                # 如果特殊棋子更接近目标区域，给予额外加分
                special_piece_bonus += 50 - self.distance_to_goal(pos, player, inboard)  # 增加加分值至20

        # 总评估值
        eval = 0.1*opponent_score - player_score + special_piece_bonus - special_penalty - endpoint_penalty
        return eval

    def distance_to_goal(self, pos, player, board):
        # 计算当前位置到达目标区域的距离（使用曼哈顿距离）
        if player == 2:
            # 玩家2的目标区域在棋盘的底部
            goal_row = board.size * 2 - 1
            return abs(pos[0] - goal_row)
        elif player == 1:
            # 玩家1的目标区域在棋盘的顶部
            goal_row = 1
            return abs(pos[0] - goal_row)
        elif player == 4:
            goal_row = board.size * 2 - 2
            if (pos[0] - goal_row) > 0:
                return 150
            else:
                return abs(pos[0] - goal_row)
        else:
            goal_row = 2
            if (pos[0] - goal_row) < 0:
                return 150
            else:
                return abs(pos[0] - goal_row)


    def is_backward_move(self, action, player):
        # 判断动作是否为后退移动
        start_pos, end_pos = action
        if player == 1:
            return end_pos[0] < start_pos[0]
        else:
            return end_pos[0] > start_pos[0]

