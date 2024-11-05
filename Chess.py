import time
import tkinter as tk
from tkinter import messagebox
import queue as q
import numpy as np
from collections import deque
import copy


class Chess():
    def __init__(self, show_popups=False, board = None, player = 'w'):
        self.is_terminated = False
        self.selection = q.Queue()
        self.rendering = False
        self.show_popups = show_popups
        self.reset() # reset board, score, player, selection
        if board is not None:
            self.board = board
            self.player = player
        self.points_table = {'_': 0, 'P': 1, 'B': 3, 'N': 3, 'R': 5, 'Q': 9}
        self.last_6_moves_by_white = deque(maxlen=6)
        self.last_6_moves_by_black = deque(maxlen=6)

        self.white_moves_without_capture_or_pawn_movement = 0
        self.black_moves_without_capture_or_pawn_movement = 0
        

    def render(self):
        # Create the tkinter window
        root = tk.Tk()
        root.title("Chess UI")

        # Piece symbols for display
        self.piece_symbols = {
            'Pw': '♙', 'Bw': '♗', 'Nw': '♘', 'Rw': '♖', 'Qw': '♕', 'Kw': '♔',
            'Pb': '♟', 'Bb': '♝', 'Nb': '♞', 'Rb': '♜', 'Qb': '♛', 'Kb': '♚',
            '_': ''  # Empty space
        }

        # A 2D list to hold the buttons
        self.button_board = [[None for _ in range(8)] for _ in range(8)]

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                symbol = self.piece_symbols[piece]

                # Create a button for each square
                button = tk.Button(
                    root,
                    text=symbol,
                    font=("Arial", 24),
                    width=2,
                    height=1,
                    bg="white" if (row + col) % 2 == 0 else "gray",
                    command=lambda r=row, c= col: self.__handle_button_click(r, c)
                )
                button.grid(row=row, column=col)
                self.button_board[row][col] = button  # Store button reference

        # Run the tkinter main loop
        self.rendering = True
        self.root = root

        root.protocol("WM_DELETE_WINDOW", self.stop_render)
        root.mainloop()

    def stop_render(self):
        if self.rendering:
            self.root.destroy()
            self.rendering = False

    def get_state(self):
        return 0, self.board, self.is_terminated, self.player

    def __handle_button_click(self, row, col):
        if self.selection.empty():
            # First selection could not be opponent
            if self.board[row][col][-1] != self.player:
                return
            self.selection.put((row, col))
            self.root.title("Chess UI " + ("White" if self.player == "w" else "Black") + "-" + str(self.selection.qsize()) + " squares selected")
            return
        
        # process the move request
        old_cords = self.selection.get()
        # task = threading.Thread(target=self.move, args=(old_cords, (row, col)))
        # task.start()
        self.move(old_cords, (row, col))
        self.root.title("Chess UI " + ("White" if self.player == "w" else "Black") + "-" + str(self.selection.qsize()) + " squares selected")
        self.selection.task_done()

    def __update_board(self):
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                symbol = self.piece_symbols[piece]
                self.button_board[row][col].config(text=symbol)

    def __is_empty(self, cords, board=None):
        if board == None:
            board = self.board
        row, col = cords
        if board[row][col] == '_':
            return True
        return False
    

    def __is_valid_cords(self, cords):
        row, col = cords
        if row < 0 or row > 7:
            return False
        if col < 0 or col > 7:
            return False
        return True
    
    def __validate_rook(self, old_cords, new_cords, board=None):
        # Todo:
        # Error: The King is unaable to identify check by a rook sometimes
        # The piece checking is ending after 1 check ig
        if board == None:
            board = self.board
        o_row, o_col = old_cords
        n_row, n_col = new_cords

        if o_col == n_col and o_row != n_row:

            if n_row > o_row:
                i = o_row + 1
                while i < n_row : # if the last square has a piece of opponent then the move is valid (hence < is used instead of <=)
                    if self.__is_empty((i, n_col), board) == False:
                        # print("\n\n\n\n" + board[i][n_col] + "\n\n\n\n")
                        return False, board[i][n_col]
                    i += 1
                
            else:
                i = o_row - 1
                while i > n_row :
                    if self.__is_empty((i, n_col), board) == False:
                        # print("\n\n\n\n" + board[i][n_col] + "\n\n\n\n")
                        return False, board[i][n_col]
                    i -= 1

            return True, board[n_row][n_col]
            
        elif o_row == n_row and o_col != n_col:

            if n_col > o_col:
                i = o_col + 1
                while i < n_col : # if the last square has a piece of opponent then the move is valid (hence < is used instead of <=)
                    if self.__is_empty((n_row, i), board) == False:
                        # print("\n\n\n\n" + board[n_row][i] + "\n\n\n\n")
                        return False, board[n_row][i]
                    i += 1
                
            else:
                i = o_col - 1
                while i > n_col :
                    if self.__is_empty((n_row, i), board) == False:
                        # print("\n\n\n\n" + board[n_row][i] + "\n\n\n\n")
                        return False, board[n_row][i]
                    i -= 1
                        
            return True, board[n_row][n_col]
            
        return False, board[n_row][n_col]

    def __validate_bishop(self, old_cords, new_cords, board=None):
        if board == None:
            board = self.board
        o_row, o_col = old_cords
        n_row, n_col = new_cords


        # Check if it's a diagonal move
        if abs(o_col - n_col) != abs(o_row - n_row):
            # print("Bishop error: unequal rows and columns")
            return False, board[n_row][n_col]
            
        # Check if the path is empty
        i, j = o_row, o_col

        # For the 1st quadrant
        if n_col > o_col and n_row < o_row:
            i -= 1
            j += 1
            while i > n_row and j < n_col:
                if self.__is_empty((i, j), board) == False:
                    # print("Bishop error: Something is blocking in 1st quadrant")
                    return False, board[i][j]
                i -= 1
                j += 1
        # For the 2nd quadrant
        elif n_col < o_col and n_row < o_row:
            i -= 1
            j -= 1
            while i > n_row and j > n_col:
                if self.__is_empty((i, j), board) == False:
                    # print("Bishop error: Something is blocking in 2nd quadrant")
                    return False, board[i][j]
                i -= 1
                j -= 1
        # For the 3rd quadrant
        elif n_col < o_col and n_row > o_row:
            i += 1
            j -= 1
            while i < n_row and j > n_col:
                if self.__is_empty((i, j), board) == False:
                    # print("Bishop error: Something is blocking in 3rd quadrant")
                    return False, board[i][j]
                i += 1
                j -= 1
        # For the 4th quadrant
        elif n_col > o_col and n_row > o_row:
            i += 1
            j += 1
            while i < n_row and j < n_col:
                if self.__is_empty((i, j), board) == False:
                    # print("Bishop error: Something is blocking in 4th quadrant")
                    return False, board[i][j]
                i += 1
                j += 1
        # print("Bishop info: All good; can move")
        return True, board[n_row][n_col]
    
    def get_opponent(self, player=None):
        if player == None:
            player = self.player
        if player == 'w':
            return 'b'
        return 'w'

    def _is_check(self, board=None, player=None):
        if board == None:
            board = self.board
        if player == None:
            player = self.player

        row, col = -1, -1

        for i in range(0, 8):
            for j in range(0, 8):
                if board[i][j] == 'K' + player:
                    row, col = i, j

        op = self.get_opponent(player)
        # Check for a rook or rook queen giving check
        rook_queen_checklist = ['R'+ op, 'Q' + op]


        if (self.__validate_rook((row, col), (row, 7), board)[1] in rook_queen_checklist) or (self.__validate_rook((row, col), (row, 0), board)[1] in rook_queen_checklist) or (self.__validate_rook((row, col), (0, col), board)[1] in rook_queen_checklist) or (self.__validate_rook((row, col), (7, col), board)[1] in rook_queen_checklist):
            return True
        
        
        

        # Checking for a bishop or bishop queen giving check
        bishop_queen_checklist = ('B' + op, 'Q' + op)

        bishop_queen_cords = []

        # print("Is check function")
        for i in (-1, 1):
            for j in (-1, 1):
                k, l = row, col
                while self.__is_valid_cords((k, l)):
                    k += i
                    l += j
                bishop_queen_cords.append((k-i, l-j))

        # print("Sending from is_check()")
        for cords in bishop_queen_cords:
            if self.__validate_bishop((row, col), cords, board)[1] in bishop_queen_checklist:
                return True
        # print("Closing from is_check()")
        # Check for a pawn giving check
        offset = 1 if player == 'b' else -1
        if (self.__is_valid_cords((row + offset, col - 1)) and board[row + offset][col - 1] == 'P' + op) or (self.__is_valid_cords((row + offset, col + 1)) and board[row + offset][col + 1] == 'P' + op):
                return True
        
        # Checking for a king giving check
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if self.__is_valid_cords((row+d_row, col+d_col)) == False or (d_row, d_col) == (0, 0):
                    continue

                if board[row+d_row][col+d_col] == 'K' + op:
                    return True
                
        # Checking for a knight giving check
        for d_row, d_col in ((2, 1), (1, 2), (-2, 1), (-1, 2), (-2, -1), (-1, -2), (1, -2), (2, -1)):
            if self.__is_valid_cords((row+d_row, col+d_col)) == False:
                continue

            if board[row+d_row][col+d_col] == 'N' + op:
                return True
            
        # If everything is alright
        return False

    def is_checkmate(self, player=None, board=None):
        if player == None:
            player = self.player
        if board == None:
            board = self.board

        for row in range(8):
            for col in range(8):
                if board[row][col][-1] == player:
                    # try placing it to every other square and check for removed check
                    for r in range(8):
                        for c in range(8):
                            if r == row and c == col:
                                continue
                            if self.is_valid_move((row, col), (r, c), player):
                                temp_board = copy.deepcopy(board)
                                temp_board[r][c] = temp_board[row][col]
                                temp_board[row][col] = '_'

                                if self._is_check(temp_board, player) == False:
                                    return False

        return True

    def is_stalemate(self, player=None):
        if player == None:
            player = self.player

        is_there_any_legal_action = self.get_legal_actions(player=player, affirm=True)

        if not is_there_any_legal_action and not self._is_check(player=player):
            return True

        return False
    
    def check_for_3_fold_repetiton(self, player=None):
        if player == None:
            player = self.player

        last_6 = list(self.last_6_moves_by_white if player == 'w' else self.last_6_moves_by_black)[::-1]
        if len(last_6) < 6:
            return False
        
        if last_6[0] == last_6[2] == last_6[4] and last_6[1] == last_6[3] == last_6[5]:
            return True
        return False

    def check_for_50_move_rule(self, player=None):
        if player == None:
            player = self.player
        board = self.board

        if player == 'w':
            return self.white_moves_without_capture_or_pawn_movement == 50
        
        return self.black_moves_without_capture_or_pawn_movement == 50

    def check_for_insufficient_material(self, board=None):
        if board == None:
            board = self.board

        white = []
        black = []

        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if piece == '_' or piece[0] == 'K':
                    continue
                sq_color = "W" if (i + j) % 2 == 0 else "B"
                if piece[-1] == 'w':
                    white.append(piece[0] + sq_color)
                else:
                    black.append(piece[0] + sq_color)

        # King vs King
        if len(white) == len(black) == 0:
            return True
        # King and Bishop/Knight vs King
        if len(white) == 0 and len(black) == 1:
            if black[0][0] == 'B' or black[0][0] == 'N':
                return True
            
        if len(white) == 1 and len(black) == 0:
            if white[0][0] == 'B' or white[0][0] == 'N':
                return True
        # King and Bishop vs King and Bishop of same color
        if len(white) == len(black) == 1:
            if white[0] == black[0]:
                return True
        
        # Sufficient Material
        return False


    def is_draw(self):
        # Draw by Stalemate
        if self.is_stalemate():
            return True
        
        # Draw by threefold repetiton
        if self.check_for_3_fold_repetiton(player=self.get_opponent()):
            return True

        # Draw by Fifty move rule
        if self.check_for_50_move_rule(player=self.get_opponent()):
            return True

        # Draw by insufficient material
        if self.check_for_insufficient_material():
            return True

        return False

    @staticmethod  
    def encode_action(old_cords, new_cords):
        o_row, o_col = old_cords
        n_row, n_col = new_cords

        from_square = o_row * 8 + o_col % 8
        to_square = n_row * 8 + n_col % 8

        return from_square * 64 + to_square % 64

    @staticmethod
    def decode_action(action):
        from_square, to_square = action // 64, action % 64
        o_row, o_col = from_square // 8, from_square % 8
        n_row, n_col = to_square // 8, to_square % 8

        return (o_row, o_col), (n_row, n_col)
    
    def get_illegal_mask(self, board=None, player=None):
        if player == None:
            player = self.player
        if board == None:
            board = self.board
        
        mask = np.zeros(64*64, dtype=np.float32)

        for o_row in range(8):
            for o_col in range(8):
                for n_row in range(8):
                    for n_col in range(8):
                        if o_row == n_row and o_col == n_col:
                            continue
                        if board[o_row][o_col] == '_' or board[o_row][o_col][-1] != player or board[n_row][n_col][-1] == player:
                            continue

                        if self.is_valid_move((o_row, o_col), (n_row, n_col), player=player, board=board):
                            
                            # making the move in a dummy env to evaluate _is_check()
                            temp_board = copy.deepcopy(self.board)

                            captured = temp_board[n_row][n_col]
                            temp_board[n_row][n_col] = temp_board[o_row][o_col]
                            temp_board[o_row][o_col] = '_'

                            if not self._is_check(temp_board, player):
                                action = self.encode_action((o_row, o_col), (n_row, n_col))
                                mask[action] = 1.0

        return mask
    

    def get_legal_actions(self, board=None, player=None, affirm=False):
        if player == None:
            player = self.player
        if board == None:
            board = self.board      

        res = []

        for o_row in range(8):
            for o_col in range(8):
                for n_row in range(8):
                    for n_col in range(8):
                        if o_row == n_row and o_col == n_col:
                            continue
                        if board[o_row][o_col] == '_' or board[o_row][o_col][-1] != player or board[n_row][n_col][-1] == player:
                            continue

                        if self.is_valid_move((o_row, o_col), (n_row, n_col), player=player, board=board):
                            
                            # making the move in a dummy env to evaluate _is_check()
                            temp_board = copy.deepcopy(board)

                            captured = temp_board[n_row][n_col]
                            temp_board[n_row][n_col] = temp_board[o_row][o_col]
                            temp_board[o_row][o_col] = '_'

                            if not self._is_check(temp_board, player):
                                if affirm:
                                    return True
                                action = self.encode_action((o_row, o_col), (n_row, n_col))
                                res.append(action)
        if affirm:
            return False  
        return res

        
    # This function does not consider geting checked after the move
    # That would be checked inside the move function
    def is_valid_move(self, old_cords, new_cords, player=None, board = None):
        if player == None:
            player = self.player
        if board == None:
            board = self.board
        
        # You can't move something that doesn't exist
        o_row, o_col = old_cords
        n_row, n_col = new_cords

        piece = board[o_row][o_col][0]
        

        # Ensuring every cordinate we deal in this function is valid
        if not self.__is_valid_cords(old_cords) or not self.__is_valid_cords(new_cords):
            return False

        # print("valid move till now", "old:", o_row, "new", n_row)
        if self.__is_empty(old_cords):
            return False
        

        # You can only move the piece that is yours
        if player != self.board[o_row][o_col][-1]:
            return False

        # You can't move to a square where there is already your piece
        if player == self.board[n_row][n_col][-1]:
            return False
        

        if piece == 'P' and player == 'b': # Pawn
            # Cannot move backwards
            if o_row >= n_row:
                return False
            
            # moving two steps if at initial position and no obstruction
            if o_row == 1 and o_row + 2 == n_row and o_col == n_col and self.__is_empty((o_row + 1, o_col)) and self.__is_empty(new_cords):
                return True
            
            # moving one step if no obstruction
            if o_col == n_col and o_row + 1 == n_row and self.__is_empty(new_cords):
                return True
            
            # Capturing a piece if diagonally one step ahead
            if self.__is_empty(new_cords) == False and o_row + 1 == n_row and (o_col + 1 == n_col or o_col -1 == n_col):
                return True
            
            # Todo: Enpassant
            # Todo: Piece upgrade
            return False
        
        
        if piece == 'P' and player == 'w': # Pawn
            # Cannot move backwards
            
            
            if o_row <= n_row:
                return False
            
            # moving two steps if at initial position and no obstruction
            if o_row == 6 and o_row - 2 == n_row and o_col == n_col and self.__is_empty((o_row-1, o_col)) and self.__is_empty(new_cords):
                return True
            
            # moving one step if no obstruction
            if o_col == n_col and o_row - 1 == n_row and self.__is_empty(new_cords):
                return True
            
            # Capturing a piece if diagonally one step ahead
            if self.__is_empty(new_cords) == False and o_row - 1 == n_row and (o_col + 1 == n_col or o_col - 1 == n_col):
                return True
            
            # Todo: Enpassant
            # Todo: Piece upgrade
            return False
        
        
        # Todo:
        # 1. Rook (done)
        # 2. Bishop (done)
        # 3. King (done)
        # 4. Queen (done)
        # 5. Knight (done)
        # 6. is_check
        # 7. is_checkmate

        if piece == 'R':
            return self.__validate_rook(old_cords, new_cords)[0]
        if piece == 'B':
            return self.__validate_bishop(old_cords, new_cords)[0]
        
        if piece == 'K':
            # move one step
            if abs(o_col - n_col) == 1 and abs(o_row - n_row) == 1:
                return True
            if abs(o_col - n_col) == 0 and abs(o_row - n_row) == 1:
                return True
            if abs(o_col - n_col) == 1 and abs(o_row - n_row) == 0:
                return True
            return False 
        
        if piece == 'Q':
            if self.__validate_rook(old_cords, new_cords)[0] or self.__validate_bishop(old_cords, new_cords)[0]:
                return True
            return False
        
        if piece == 'N':
            dx = abs(n_col - o_col)
            dy = abs(n_row - o_row)

            if dx == dy:
                return False

            if (dx == 1 or dx == 2) and (dy == 1 or dy == 2):
                return True
            return False

    def reset(self):
        self.is_terminated = False
        self.player = 'w'
        self.score_white = 0
        self.score_black = 0
        self.board = [
            ['Rb', 'Nb', 'Bb', 'Qb', 'Kb', 'Bb', 'Nb', 'Rb'],
            ['Pb', 'Pb', 'Pb', 'Pb', 'Pb', 'Pb', 'Pb', 'Pb'],
            ['_', '_', '_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_', '_', '_'],
            ['Pw', 'Pw', 'Pw', 'Pw', 'Pw', 'Pw', 'Pw', 'Pw'],
            ['Rw', 'Nw', 'Bw', 'Qw', 'Kw', 'Bw', 'Nw', 'Rw'],
        ]
        while(self.selection.empty() == False):
            self.selection.get()
            self.selection.task_done()

        return 0, self.board, False, self.player


    # it is the action function of this env
    # outputs-> reward, new state, is_terminated, new player     
    def move(self, old_cords, new_cords):
        o_row, o_col = old_cords
        n_row, n_col = new_cords


        if self.is_valid_move(old_cords, new_cords, self.player) == False:
            # self.player = self.get_opponent(self.player)
            # penalizing for choosing a invalid move
            return -2, self.board, False, self.player
        
        
        temp_board = copy.deepcopy(self.board)

        captured = temp_board[n_row][n_col]
        temp_board[n_row][n_col] = temp_board[o_row][o_col]
        temp_board[o_row][o_col] = '_'


        # print("HI there")
        if self._is_check(temp_board, self.player):
            print("Invalid move. Your King will be on check")
            # self.player = self.get_opponent(self.player)
            return -2, self.board, False, self.player
    
        
        
        if self.player == 'w':
            self.score_white += self.points_table[captured[0]]
            self.score_black -= self.points_table[captured[0]]

            # For 3 fold repetition
            self.last_6_moves_by_white.append((o_row, o_col, n_row, n_col))

            # For 50 moves rule
            if captured[0] == '_' and temp_board[n_row][n_col][0] != 'P':
                self.white_moves_without_capture_or_pawn_movement += 1
            else:
                self.white_moves_without_capture_or_pawn_movement = 0
        else:
            self.score_black += self.points_table[captured[0]]
            self.score_white -= self.points_table[captured[0]]

            # For 3 fold repetition
            self.last_6_moves_by_black.append((o_row, o_col, n_row, n_col))

            # For 50 moves rule
            if captured[0] == '_' and temp_board[n_row][n_col][0] != 'P':
                self.black_moves_without_capture_or_pawn_movement += 1
            else:
                self.black_moves_without_capture_or_pawn_movement = 0

        self.board = temp_board
        if self.rendering:
            self.__update_board()

        if self._is_check(temp_board, self.get_opponent(self.player)):
            # If you have checked the other player, check for checkmate
            if self.is_checkmate(player = self.get_opponent(self.player)):
                self.is_terminated = True
                if self.rendering:
                    if self.show_popups:
                        messagebox.showinfo("Info", ("White " if self.player == 'b' else "Black ") + "got Checkmate!!" )
                    self.root.destroy()
                    self.rendering = False
                return self.points_table[captured[0]] + 95, self.board, True, self.get_opponent(self.player)

        self.player = self.get_opponent(self.player)

        if self.is_draw():
            self.is_terminated = True
            if self.rendering:
                    if self.show_popups:
                        messagebox.showinfo("Info", ("White " if self.player == 'b' else "Black ") + "got Checkmate!!" )
                    self.root.destroy()
                    self.rendering = False
            return 0.5, self.board, True, self.player

        # Switch player

        return (self.points_table[captured[0]] if captured[0] != '_' else 1), self.board, False, self.player
        

if __name__ == '__main__':
    chess = Chess()
    t1 = time.time()
    # get_legal_actions(chess.board, chess.player)
    chess.get_legal_actions()
    t2 = time.time()
    print(t2-t1)
