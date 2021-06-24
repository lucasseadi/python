# -*- coding: utf-8 -*-

# Hangman Game (Jogo da Forca)
# Programação Orientada a Objetos

# Import
import random

# Board (tabuleiro)
board = ['''

>>>>>>>>>>Hangman<<<<<<<<<<

+---+
|   |
    |
    |
    |
    |
=========''', '''

+---+
|   |
O   |
    |
    |
    |
=========''', '''

+---+
|   |
O   |
|   |
    |
    |
=========''', '''

 +---+
 |   |
 O   |
/|   |
     |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
     |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
/    |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
/ \  |
     |
=========''']


# Classe
class Hangman:

	# Método Construtor
	def __init__(self, word):
		self.word = word
		self.tentativas = 0
		self.erros = 0
		self.letras_erradas = []
		self.letras_corretas = []

	# Método para adivinhar a letra
	def guess(self, letter):
		if letter in self.word and letter not in self.letras_corretas:
			self.letras_corretas.append(letter)
		elif letter not in self.word and letter not in self.letras_erradas:
			self.letras_erradas.append(letter)
			self.erros += 1

	# Método para verificar se o jogo terminou
	def hangman_over(self):
		return self.erros == 6 or self.hangman_won()

	# Método para verificar se o jogador venceu
	def hangman_won(self):
		venceu = True
		for letter in self.word:
			if letter not in self.letras_corretas:
				venceu = False
				break
		return venceu

	# Método para não mostrar a letra no board
	def hide_word(self):
		word = list(map(lambda letra : letra if letra in self.letras_corretas else '_', self.word))
		return ''.join(map(str, word))

	# Método para checar o status do game e imprimir o board na tela
	def print_game_status(self):
		print(board[self.erros])
		print('Palavra: %s' % (self.hide_word()))
		print('Letras erradas: \n%s' % '\n'.join(self.letras_erradas))
		print('Letras corretas: \n%s' % '\n'.join(self.letras_corretas))


# Função para ler uma palavra de forma aleatória do banco de palavras
def rand_word():
	with open("palavras.txt", "rt") as f:
		bank = f.readlines()
		return bank[random.randint(0, len(bank))].strip()


# Função Main - Execução do Programa
def main():

	# Objeto
	game = Hangman(rand_word())

	# Enquanto o jogo não tiver terminado, print do status, solicita uma letra e faz a leitura do caracter
	while not game.hangman_over():
		# Verifica o status do jogo
		game.print_game_status()
		#game.guess(input("Digite uma letra: "))
		game.guess(input("Digite uma letra: "))
		game.tentativas += 1

	# De acordo com o status, imprime mensagem na tela para o usuário
	if game.hangman_won():
		game.print_game_status()
		print ('\nParabéns! Você venceu!!')
	else:
		game.print_game_status()
		print ('\nGame over! Você perdeu.')
		print ('A palavra era ' + game.word)
		
	print ('\nFoi bom jogar com você! Agora vá estudar!\n')


# Executa o programa		
if __name__ == "__main__":
	main()