import random
import numpy as np
import cv2
from PIL import Image
from time import sleep

class tetris:
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (0, 240, 240),#I
        2: (160, 0, 240),#T
        3: (240, 160, 0),#L
        4: (0, 0, 240),#J
        5: (240, 0, 0),#Z
        6: (0, 240, 0),#S
        7: (240, 240, 0)#o
    }

    def __init__(self, end_score):
        self.reset()
        self.end_score = end_score

    def reset(self): #newgame
        self.board = [[0]*tetris.BOARD_WIDTH for _ in range(tetris.BOARD_HEIGHT)]
        self.score = 0
        self.game_over = False
        self.waiting_queue = list(range(len(tetris.TETROMINOS)))
        random.shuffle(self.waiting_queue)
        self.line_cleared = 0
        self.current_pos = [3, 0]
        self.current_rotation = 0
        self.current_block = self.waiting_queue.pop()

    def play(self, x, rotation, render = False):
        self.current_pos = [x, 3]
        self.current_rotation = rotation
        
        while not self._check_error(self._rotate_block(rotation, self.current_block), self.current_pos, self.board):
            if render:
                self.render(self._add_block(self._rotate_block(rotation, self.current_block), self.current_pos, self.current_block))
                sleep(0.01)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1
        
        self.board = self._add_block(self._rotate_block(rotation, self.current_block), self.current_pos, self.current_block)
        line_cleared, self.board = self._clear_lines(self.board)

        score, game_over = self._get_reward_check_if_end(line_cleared, self.board)

        if not game_over:
            self._next_round()
        
        self.score += score

        if self.score >= self.end_score:
            game_over = True

        self.game_over = game_over

        return score, self.game_over

    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_block
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_error(piece, pos, self.board):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_block(piece, pos, piece_id)
                    states[(x, rotation)] = self._do_state(board)

        return states

    def _get_reward_check_if_end(self, line_cleared, board):
        self._check_queue()

        score = 1 + (line_cleared ** 2) * tetris.BOARD_WIDTH
        game_over = False

        if self._check_error(self._rotate_block(0, self.waiting_queue[0]), [3, 0], board):
            game_over = True
        
        if game_over:
            score -= 2

        return score, game_over 

    def _do_state(self, board)->tuple[list[int], int, bool]:
        line, board = self._clear_lines(board)
        hole = self._get_hole(board)
        bump = self._get_bump(board)
        total_height = self._get_total_height(board)
        score, gameover = self._get_reward_check_if_end(line, board)
        return ([line, hole, bump, total_height], score, gameover)

    def _check_queue(self):
        if len(self.waiting_queue) > 0:
            return
        adding_queue = list(range(len(tetris.TETROMINOS)))
        random.shuffle(adding_queue)
        for block in adding_queue:
            self.waiting_queue.append(block)
        return

    def _get_height(self, x, board):
        height = tetris.BOARD_HEIGHT
        for i in range(tetris.BOARD_HEIGHT):
            if board[i][x] == 0:
                height -= 1
            else: 
                break
        return height

    def _get_empty_count(self, board):
        count = 0
        for i in range(tetris.BOARD_WIDTH):
            if self._get_height(i, board) == 0:
                count += 1
        return count

    def _get_total_height(self, board):
        height = 0
        for i in range(tetris.BOARD_WIDTH):
            height += self._get_height(i, board)
        return height

    def _get_hole(self, board):
        hole = 0
        for j in range(tetris.BOARD_WIDTH):
            current_hole = 0
            flag = 0
            for i in range(tetris.BOARD_HEIGHT):
                if board[i][j] == 0 and flag == 1:
                    current_hole += 1
                elif board[i][j] >= 1: 
                    flag = 1
            hole += current_hole
        return hole
    
    def _get_bump(self, board):
        bump = 0
        tmp = self._get_height(0, board)
        for i in range(1, tetris.BOARD_WIDTH):
            tmp2 = self._get_height(i, board)
            bump += abs(tmp - tmp2)
            tmp = tmp2
        return bump
    
    def _get_max_bump(self, board):
        max_bump = 0
        tmp = self._get_height(0, board)
        for i in range(1, tetris.BOARD_WIDTH):
            tmp2 = self._get_height(i, board)
            max_bump = max(max_bump, abs(tmp - tmp2))
            tmp = tmp2
        return max_bump
    
    def _clear_lines(self, board):
        new_board = []
        clear = 0
        for x in range(tetris.BOARD_HEIGHT):
            block_cnt = 0
            for y in range(tetris.BOARD_WIDTH):
                if board[x][y] > 0:
                    block_cnt += 1
            if block_cnt != tetris.BOARD_WIDTH:
                new_board.append(board[x])
            else:
                clear += 1
        for _ in range(clear):
            new_board.insert(0, [0 for _ in range(tetris.BOARD_WIDTH)])
        return clear, new_board

        """ clear = [idx for idx, row in enumerate(board) if sum(row) == tetris.BOARD_WIDTH]
        if clear:
            board = [row for idx, row in enumerate(board) if idx not in clear]
            for _ in clear:
                board.insert(0, [0 for _ in range(tetris.BOARD_WIDTH)])
        return len(clear), board """

    def _rotate_block(self, angle, block):
        return tetris.TETROMINOS[block][angle]

    def _add_block(self, block, pos, block_id):       
        board = [x[:] for x in self.board]
        for x, y in block:
            board[y + pos[1]][x + pos[0]] = block_id+1
        return board
    
    def _check_error(self, block, pos, board):
        for x, y in block:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= tetris.BOARD_WIDTH \
                    or y < 0 or y >= tetris.BOARD_HEIGHT \
                    or board[y][x] >= 1:
                return True
        return False
    
    def _next_round(self):
        self._check_queue()
        
        self.current_block = self.waiting_queue.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

    def render(self, borad):
        img = [tetris.COLORS[p] for row in borad for p in row]
        img = np.array(img).reshape(tetris.BOARD_HEIGHT, tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((tetris.BOARD_WIDTH * 25, tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)
