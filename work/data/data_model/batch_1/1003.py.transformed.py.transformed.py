import pygame
import tkinter
import math
pygame.init()
num = 0
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
MAUVE = (224, 176, 255)
clock = pygame.time.Clock()
def get_num(m, n):
    global num
    num = int(n.get())
    m.destroy()
def text_objects(text, font):
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()
def message_display(text, x, y, size):
    largeText = pygame.font.Font('freesansbold.ttf', size)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = (x, y)
    display.blit(TextSurf, TextRect)
    pygame.display.update()
class Cell:
    def __init__(self, x, y, n):
        self.pos = (x, y)
        self.value = n
        self.highlighted = False
        celldict[self.value] = self
        message_display(f"{self.value}", x + 16, y + 16, 17)
    def highlight(self):
        if not self.highlighted:
            x, y = self.pos
            pygame.draw.rect(display, MAUVE, (x, y, 32, 32))
            message_display(f"{self.value}", x + 16, y + 16, 17)
            self.highlighted = True
            pygame.time.wait(50)
m = tkinter.Tk()
l1 = tkinter.Label(m, text='Enter a number', width=25)
n = tkinter.Entry(m)
btn = tkinter.Button(m, text='Calculate', command=lambda: get_num(m, n))
l1.pack()
n.pack()
btn.pack()
m.mainloop()
sq_num = num
if (math.floor(math.sqrt(num))**2 - num != 0):
    sq_num = (math.floor(math.sqrt(num)) + 1)**2
dim = int(sq_num**0.5)
width = 32 * dim
display = pygame.display.set_mode((width, width))
display.fill(WHITE)
for i in range(0, width, 32):
    pygame.draw.line(display, BLACK, (i, 0), (i, width)
                     )
for j in range(0, width, 32):
    pygame.draw.line(display, BLACK, (0, j), (width, j))
pygame.display.update()
celldict = {}
count = 1
for i in range(0, width, 32):
    pygame.event.pump()
    for j in range(0, width, 32):
        if (count <= num):
            cell = Cell(j, i, count)
            clock.tick()
            count += 1
        else:
            break
pygame.display.update()
nums = set(range(2, num + 1))
composites = {1}
celldict[1].highlight()
for i in range(
        2,
        round(
            math.sqrt(num)) +
        1):
    if i not in composites:
        comp = [i * k for k in range(2, num // i + 1)]
        composites.update(comp)
        for val in comp:
            pygame.event.pump()
            celldict[val].highlight()
primetxt = ''
for val, cell in celldict.items():
    pygame.event.pump()
    if not cell.highlighted:
        x, y = cell.pos
        pygame.draw.rect(display, GREEN, (x, y, 32, 32))
        message_display(f"{val}", x + 16, y + 16, 17)
        pygame.time.wait(100)
        primetxt += str(val) + ' ,'
m = tkinter.Tk()
l1 = tkinter.Label(m, text='Primes that were sieved: ')
msg = tkinter.Message(m, text=primetxt[:-1], width=200)
l1.pack()
msg.pack()
m.mainloop()
