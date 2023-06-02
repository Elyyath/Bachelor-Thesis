import pygame
from freegames import vector, floor
import numpy as np
import time
import tensorflow as tf

WINDOW = pygame.display.set_mode([1400,700])


class Environment():
    def __init__(self, worker_id):
        self.tiles = {
            0: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            1: [0,2,2,2,0,0,0,0,0,0,0,0,1,1,4,4,4,0],
            2: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            3: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            4: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            5: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            6: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            7: [0,2,2,2,1,1,0,0,0,0,0,0,0,0,4,4,4,0],
            8: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        }
        self.player = vector(50,80)
        self.action_space = ['up', 'down', 'left', 'right']
        self.id = worker_id
        self.surface_position = self.get_surface_position()
        self.balls = [
            [vector(110, 50),vector(4,0)],
            [vector(250, 70),vector(-4,0)],
            [vector(110, 90),vector(4,0)],
            [vector(250, 110),vector(-4,0)],
            [vector(110, 130),vector(4,0)]
        ]

        self.done = False
        #weil pygame keine multiplen Fenster unterstützt, werden 16 einzelne Oberflächen in einem Fenster dargestellt
        self.screen  = pygame.Surface((350,175))
        self.current_screen = None
        self.createMap()
        
    #weist jedem Thread eine Position im Fenster zu
    def get_surface_position(self):
        
        x = self.id % 4 * 350
        y = int(self.id/4) *175
        
        return vector(x,y)
        
    def close(self):
        pygame.quit()
        
    def num_actions_available(self):
        return len(self.action_space)
    
    #zeichnet das Level auf Basis der Tilemap
    def createMap(self):
        for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
        
        for row in range(len(self.tiles)):
            for col in range(len(self.tiles[row])):
                if self.tiles[row][col] >0:
                    if self.tiles[row][col] % 2 == 0:
                        color= (76, 230, 41)
                    else:
                        if (row % 2 == 0 and col % 2 == 1) or (row % 2 == 1 and col % 2 == 0):
                            color=(156, 156, 156)
                        else:
                            color = (255, 255, 255)

                    xpos = col * 20
                    ypos = row * 20
                    pygame.draw.rect(self.screen, color ,(xpos,ypos, 20,20))


        self.createObjects()
        WINDOW.blit(self.screen, self.surface_position)
        pygame.display.update()
    
    #zeichnet die Spielfigur und die Bälle
    def createObjects(self):

        for ball, course in self.balls:
            pygame.draw.circle(self.screen, (0,0,255),ball, 5)

        pygame.draw.rect(self.screen, (255,0,0), (self.player.x,self.player.y, 10,10))
        
    
    def reset(self):
        self.player = vector(50,80)
        self.balls = [
            [vector(110, 50),vector(4,0)],
            [vector(250, 70),vector(-4,0)],
            [vector(110, 90),vector(4,0)],
            [vector(250, 110),vector(-4,0)],
            [vector(110, 130),vector(4,0)]
        ]
        self.done = False
        self.current_screen = None
        self.createMap()
        
        return self.get_state()
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = tf.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_processed_screen(self):
        
        screen = pygame.surfarray.array3d(self.screen).transpose((1,0,2)) #tensorflow erwartet (height, width, channel)
        return self.transform_screen_data(screen)

    #Bringt die Bilddaten in die vom Netzwerk erwartete Form
    def transform_screen_data(self, screen):       
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = tf.convert_to_tensor(screen, dtype=tf.float32)
        screen = screen[tf.newaxis] #fügt eine Batch Dimension hinzu (batch, height, width, channel)
        screen = tf.image.resize(screen, [20,45])
        return screen
    
    #ermittelt, auf welchem Tile die Spielobjekte sich befinden
    def offset(self, position):
        row = int(floor(position.y, 20)/20)
        col = int((floor(position.x, 20)/20))
        return row, col

    #Funktion, die für das Feedback der Umgebung zuständig ist
    #berechnet die Höhe der Belohnung und prüft, ob die gewählte Aktion ausgeführt werden kann
    #prüft auch auf Game-Over
    def validate(self, position):
        
        valid_move = True
        done = False
        reward = 0
        
       #obere, linke Ecke
        row,col = self.offset(position)

        if self.tiles[row][col] == 0:
            valid_move = False
        
            
            
        #obere, rechte Ecke
        row, col = self.offset(vector(position.x + 9, position.y))

        if self.tiles[row][col] == 0:
            valid_move = False
            
                
        
        #untere, linke Ecke
        row, col = self.offset(vector(position.x, position.y + 9))

        if self.tiles[row][col] == 0:
            valid_move = False
            
        

        #untere, rechte Ecke        
        row, col = self.offset(position+9)

        if self.tiles[row][col] == 0:
            valid_move = False
            
        if self.tiles[row][col] == 4:
            reward = 500
            done = True
            
        if self.tiles[row][col] == 1:
            
            score =  250 - abs(position-vector(270,25)) 
            if score > reward:
                reward = score
        
        if self.tiles[row][col] == 2:
            score = 0.05 * abs(position-vector(20,20))
            if score > reward:
                reward = score
                
        return valid_move, done, reward
    
    
    #wenn False wechselt der Ball die Bewegungsrichtung  
    def validBallPosition(self, position):
        row,col = self.offset(position)

        if self.tiles[row][col] == 0:
            return False

        row, col = self.offset(position+4)

        if self.tiles[row][col] == 0:
            return False    


        row, col = self.offset(position-4)

        if self.tiles[row][col] == 0:
            return False   

        return True
    
    def collision(self):
    
        playerpos = vector(self.player.x+5,self.player.y+5)
        for ball, course in self.balls:

            if abs(ball - playerpos) < 10:
                return True

        return False
    
    
    #führt die gewählte Aktion in der Umgebung aus
    def step(self, action):
        
        if self.action_space[action] == 'up':
            direction = vector(0, 3)
            
        elif self.action_space[action] == 'down':
            direction = vector(0, -3)
            
        elif self.action_space[action] == 'left':
            direction = vector(-3, 0)
        
        else:
            direction = vector(3, 0)
        
        
        valid_move, done, reward = self.validate(self.player + direction)
        
        if valid_move:
             self.player.move(direction)

        
        for ball, course in self.balls:

            position = vector(ball.x+course.x,ball.y+course.y)
            if self.validBallPosition(position):
                ball.move(course)
            else:
                course.x = course.x*(-1)
                ball.move(course)
        
        
        if self.collision():
            done = True
            reward = -50
           

        self.screen.fill((0,0,0))    
        self.createMap()
        
        return self.get_state(), reward, done

    

