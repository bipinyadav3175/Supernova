from multiprocessing import Pool
import numpy as np
import random
import torch
from torch.nn import functional as F
from Node import Node
import operator
from Chess import Chess
from network import Model

import threading

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MCTS():
    def __init__(self, root=None, player=None, c=1.414213, model=None): # c = sqrt(2)
        if not isinstance(root, Node):
            raise ValueError("Root node is required for mcts")
        if player == None:
            raise ValueError("Please specify the player which has to make the move")
        if not isinstance(model, Model):
            raise ValueError("Invalid Model")
        
        self.root = root
        self.to_play = player
        self.c = c
        self.model = model

    def set_root(self, root):
        if not isinstance(root, Node):
            raise ValueError("Invali root node given")
        
        self.root = root

    def ucb(self, node, player):
        # calculating the ucb value for selection process
        c = self.c if player == self.to_play else -self.c
        return node.value + c * np.sqrt(np.log(node.parent.visits / (node.visits + 1))) + node.prob

    def select(self, node):
        """
        Given a node, select the child with the best value
        """

        children = [(child, self.ucb(child, node.player)) for child in node.children]

        if self.to_play == node.player:
            node, value = max(children, key=operator.itemgetter(1))
        else:
            node, value = min(children, key=operator.itemgetter(1))

        return node
    
    def selection(self):
        """
        Given a node keep selecting until to get to a leaf node with no children (or a winning state)
        """

        curr_node = self.root
        dummy = Chess()
        while len(curr_node.children) != 0 and not dummy.is_checkmate(board=curr_node.state, player=dummy.get_opponent(curr_node.player)):
            curr_node = self.select(curr_node)

        return curr_node
    
    def expansion(self, node):
        """
        Create child nodes for the given node
        """        
        
        dummy = Chess(board=node.state, player=node.player)
        if dummy.is_checkmate(player=dummy.get_opponent(node.player)):
            return node

        mask = dummy.get_illegal_mask(board=node.state, player=node.player)

        state = self.model.encode_board(node.state, node.player).unsqueeze(0)
        with torch.autocast(device_type=device):
            with torch.no_grad():
                probs, value = self.model(state, mask)
            sampled_actions = torch.multinomial(probs, num_samples=min(int(mask.sum()), 20)).squeeze(0).tolist()
        sampled_actions = [action for action in sampled_actions if mask[action]]

    
        probs = probs.squeeze(0)

        for action in sampled_actions:
            temp = Chess(board=node.state, player=node.player)
            old_cords, new_cords = Chess.decode_action(action)
            _, new_board, _, _ = temp.move(old_cords, new_cords)

            node.children.append(Node(state=new_board, player=temp.get_opponent(node.player), prob=probs[action].item(), parent=node, action=action))

        return node

    def simulation(self, node):
        """
        Take a node and simulate it using the policy or just pridict its value using critic
        """
        dummy = Chess(board=node.state, player=node.player)
        # thread = threading.Thread(target=dummy.render)
        # thread.start()
        # thread.join()
        mask = dummy.get_illegal_mask(board=node.state, player=node.player)
        state = self.model.encode_board(node.state, node.player).unsqueeze(0)
        if random.random() < self.model.eps_critic:
            with torch.autocast(device_type=device):
                with torch.no_grad():
                    _, value = self.model(state, mask)
            return value.squeeze(0).item()
        else:
            last_reward = 0
            # simulate till end
            while True:
                # sample from the actor
                with torch.autocast(device_type=device):
                    with torch.no_grad():
                        probs, _ = self.model(state, mask)
                    action = torch.multinomial(probs, num_samples=1).squeeze(0).item()
                old_cords, new_cords = Chess.decode_action(action)
                # print("Simulation action: ", old_cords, new_cords, dummy.player)
                last_reward, obs, is_terminated, pla = dummy.move(old_cords, new_cords)


                if is_terminated:
                    break

                state = self.model.encode_board(obs, dummy.player).unsqueeze(0)
                mask = dummy.get_illegal_mask(board=obs, player=dummy.player)

        winner = dummy.player
        # Draw condition
        if last_reward == 0.5:
            return 0.5
        if winner == self.to_play:
            return 1
        return 0

    @staticmethod 
    def backward(sim_node, reward):
        """
        Update the value and visits of all the nodes that lead to win/loose starting from the simulated node to the root node
        """

        node = sim_node
        node.visits += 1

        while node.parent:
            node.parent.visits += 1
            node.value += (reward - node.value) / node.visits
            node = node.parent

    def select_actual_action(self):
        """
        After simulations, select the action with higest number of visits
        (
        why not choose the node with highest value?
        cause we want the winning probability to be high.
        A node with 90 visits and 45 wins is more likely to get to the winning state that a node with 3 visits and 2 wins.
        )
        """
        children = [(child, child.visits) for child in self.root.children]
        node, value = max(children, key=operator.itemgetter(1))

        return node
    
    def get_root_distribution(self):
        """
        Returns the prob distribution (as a tensor) of the children of root node
        """
        dist = torch.zeros((4096), dtype=torch.float).to(device)
        
        for child in self.root.children:
            dist[child.action] = child.visits

        with torch.autocast(device_type=device):
            dist = F.softmax(dist, dim=0)
        return dist


        