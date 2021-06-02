from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def make_tcn(params):
    tcn_layer = TCN(input_shape=(params["time_steps"], params["input_dim"]))
    m = Sequential([tcn_layer,Dense(1)])
    m.compile(optimizer='adam', loss='mse')
    tcn_full_summary(m, expand_residual_blocks=False)

make_tcn({"time_steps":50,"input_dim":10})