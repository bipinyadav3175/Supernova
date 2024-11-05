from Chess import Chess

class Node():
    def __init__(self, state=None, player='w', prob=0, parent=None, action=None):
        if state == None:
            raise ValueError("State cannot be none")
        self.value = 0
        self.visits = 0
        self.parent = parent
        self.children = []
        self.state = state
        self.player = player
        self.prob = prob
        self.action = action

    def get_children(self):
        chess = Chess(board=self.state, player=self.player)
        legal_actions = chess.get_legal_actions()

        for action in legal_actions:
            dummy = Chess(board=self.state, player=self.player)

            old_cords, new_cords = Chess.decode_action(action)
            _, new_state, _, _ = dummy.move(old_cords, new_cords)

            self.children.append(Node(new_state, dummy.player))

        return self.children
