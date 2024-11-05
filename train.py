import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from threading import Thread
from Chess import Chess
from Node import Node
from mcts import MCTS
from network import Model
from replay_buffer import Buffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def simulate_mcts(mcts):
    t1 = time.time()
    leaf = mcts.selection()
    t2 = time.time()
    # print(f"Selection: {t2-t1}s")
    sim_node = mcts.expansion(leaf)
    t3 = time.time()
    print(f"Expansion: {t3-t2}s")
    reward = mcts.simulation(sim_node)
    t4 = time.time()
    # print(f"Simulation: {t4-t3}s")
    mcts.backward(sim_node, reward)
    t5 = time.time()
    # print(f"Backward: {t5-t4}s")

def train(optimizer, network, num_episodes=1000, num_simulations=800, batch_size=32, path="supernova-v0", graph_freq=100):
    if not isinstance(network, Model):
        raise ValueError("Valid model not found")
    
    buffer = Buffer(max_len=100_000)
    best_loss = float("inf")
    actor_log = []
    critic_log = []
    loss_log = []
    
    for i in range(num_episodes):
        t99 = time.time()
        env = Chess()
        _, board, terminated, player = env.get_state()
        root = Node(state=board, player=player)

        mcts = MCTS(root, player, model=network)

        while True:
            for _ in range(num_simulations):
                print("simulation: ", _)
                t1 = time.time()
                leaf = mcts.selection()
                t2 = time.time()
                print(f"Selection: {t2-t1}s")
                sim_node = mcts.expansion(leaf)
                t3 = time.time()
                print(f"Expansion: {t3-t2}s")
                reward = mcts.simulation(sim_node)
                t4 = time.time()
                print(f"Simulation: {t4-t3}s")
                mcts.backward(sim_node, reward)
                t5 = time.time()
                print(f"Backward: {t5-t4}s")
                print("The simulation ended")


            dist = mcts.get_root_distribution()

            # Add the case to replay buffer
            buffer.add((Model.encode_board(mcts.root.state, mcts.root.player), dist, mcts.root.value, mcts.root.player))

            # Update root node
            root = mcts.select_actual_action()
            mcts.set_root(root)

            # Take the action in the env
            old_cords, new_cords = Chess.decode_action(root.action)
            _, board, is_terminated, player = env.move(old_cords, new_cords)
            if is_terminated:
                break
        # Sample from the buffer to train the network
        batch = buffer.sample(batch_size)
        
        # Train
        obs = torch.stack([state for state, dist, reward, player in batch])
        target_dist = torch.stack([dist for state, dist, reward, player in batch])
        target_value = torch.stack([torch.tensor([reward], dtype=torch.float).to(device) for state, dist, reward, player in batch])

        mask = torch.stack([torch.tensor(env.get_illegal_mask(board=state, player=player), dtype=torch.float).to(device) for state, dist, reward, player in batch])

        with torch.autocast(device_type=device):
            pred_dist, value = network(obs, mask)

            actor_loss = torch.mean(torch.sum(-target_dist * F.log_softmax(pred_dist, dim=1), 1))
            critic_loss = F.mse_loss(value, target_value)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t100 = time.time()
        print(f'Time for this episode = {t100 - t99}')
        # Logging
        print("Episode: {} | Actor loss: {} | Critic loss: {} | Total loss: {}".format(i, actor_loss.item(), critic_loss.item(), loss.item()))
        actor_log.append(actor_loss.item())
        critic_log.append(critic_loss.item())
        loss_log.append(loss.item())

        if loss.item() < best_loss:
            torch.save(network.state_dict(), path)

        if i % graph_freq == 0:
            # Save graph as image
            episodes = list(range(1, len(loss_log) + 1))

            figure, axis = plt.subplots(2, 2)

            axis[0, 0].plot(episodes, actor_log)
            axis[0, 0].set_title("Actor loss")

            axis[0, 1].plot(episodes, critic_log)
            axis[0, 1].set_title("Critic loss")

            axis[1, 0].plot(episodes, loss_log)
            axis[1, 0].set_title("Total loss")

            plt.savefig("losses.png")

