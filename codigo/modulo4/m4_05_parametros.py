# Módulo 4 · ¿Cuántos parámetros (pesos + sesgos) tiene cada tipo de capa?
# Estas fórmulas son las que usa Keras en model.summary().

def dense(n_entrada, n_neuronas):
    return n_entrada * n_neuronas + n_neuronas            # pesos + 1 sesgo por neurona

def conv2d(k, canales_entrada, filtros):
    return (k * k * canales_entrada + 1) * filtros          # cada filtro: k*k*C pesos + 1 sesgo

def simple_rnn(n_entrada, unidades):
    return unidades * (unidades + n_entrada + 1)            # W_xh + W_hh + b

def lstm(n_entrada, unidades):
    return 4 * unidades * (unidades + n_entrada + 1)        # 4 bloques: f, i, c~, o

def gru(n_entrada, unidades):
    return 3 * unidades * (unidades + n_entrada + 2)        # 3 bloques (Keras reset_after=True: 2 sesgos)

def embedding(vocabulario, dimension):
    return vocabulario * dimension                          # una fila (vector) por palabra

print("Dense 784 -> 128              :", dense(784, 128))
print("Dense 128 -> 10               :", dense(128, 10))
print("Dense 28*28*3=2352 -> 100     :", dense(28 * 28 * 3, 100))
print("Conv2D 3x3, 1 canal, 32 filtros :", conv2d(3, 1, 32))
print("Conv2D 3x3, 3 canales, 32 filtros:", conv2d(3, 3, 32))
print("Conv2D 3x3, 32 -> 64 filtros   :", conv2d(3, 32, 64))
print("SimpleRNN entrada 8, 16 unidades:", simple_rnn(8, 16))
print("LSTM entrada 8, 16 unidades     :", lstm(8, 16))
print("GRU entrada 8, 16 unidades      :", gru(8, 16))
print("Embedding vocab 10000, dim 16   :", embedding(10000, 16))
print("\nRelación LSTM / SimpleRNN (mismas dims):", lstm(8, 16) / simple_rnn(8, 16))
