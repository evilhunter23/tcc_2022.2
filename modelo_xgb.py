from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from datetime import date, timedelta
import pandas_datareader.data as web
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import base64


st.set_page_config(page_title='Machine learning ações')
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('financial-business.png')

st.title("Projeto TCC")

arq1 = open('bolsa.txt', 'r', encoding='utf-8')


col1, col2 = st.columns([3,3])
#---------- input pricipal -------------#
sigla = col1.text_input('Digite um ticker de empresa', 'petr4.sa')
inicio = col2.date_input("Selecione uma data de inicio", date(2016,1,1))
hoje = date.today()
df = web.DataReader(sigla, 'yahoo', inicio, hoje)
tam = df.shape[0]
d1 = df.index[0]
d2 = df.index[tam-1]
d1 = d1.strftime('%d/%m/%Y')
d2 = d2.strftime('%d/%m/%Y')
ticket = sigla
texto = arq1.readline()
st.write(texto)
#---------- input pricipal -------------#


#---------- visualização 1 -------------#
def mostra_dados():
    st.subheader(f'Foram trazidos dados de {d1} até {d2}')
    st.dataframe(df.drop(columns=['Futuro']))
    arq2 = open('dataframe.txt','r', encoding='utf-8')
    t = arq2.readline()
    st.write(t)
    st.markdown("---")
#---------- visualização 1 -------------#

#---------- Grafico demonstrativo 1 -------------#
def graf_2():
    fig2 = go.Figure(data=go.Ohlc(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig2.update_layout(
        title={
        'text': f"<b>Ações {sigla}</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title='Real(R$)',
        xaxis_title='Período',
        width=760
    )
    fig2.update(layout_xaxis_rangeslider_visible=False)
    st.write(fig2)
    arq3= open('grafico1.txt', 'r', encoding='utf-8')
    t2 = arq3.readline()
    st.write(t2)
#---------- Grafico demonstrativo 1 -------------#

#---------- treinando modelo -------------#
df['Futuro'] = df['Close'].shift(-1)
df = df.drop(columns=['Adj Close'])
df['Futuro'] = df['Futuro'].fillna(0)
X = df[['High', 'Low', 'Open', 'Close', 'Volume']]
y = df[['Futuro']]


tam_treino = int(X.shape[0]*0.7)
tam_teste = int(X.shape[0]*0.3)

treino_X = X.iloc[0:tam_treino,:]
teste_X = X.iloc[tam_treino:,:]
#treino_X = treino_X.drop(columns=['Close'])
#teste_X = teste_X.drop(columns=['Close'])
treino_y = y.iloc[0:tam_treino,:]
teste_y = y.iloc[tam_treino:,:]

scaler = MinMaxScaler()
treino_X_norm = scaler.fit_transform(treino_X)
teste_X_norm = scaler.transform(teste_X)

#n = teste_X_norm.shape[0]
#amanha = teste_X_norm[n-1:]
#amanha = np.array(amanha)

modelo = xgb.XGBRegressor(n_estimators=1300, learning_rate=0.12, random_state=0, num_parallel_tree=2, max_depth=4)
modelo.fit(treino_X_norm, treino_y)
previsao = modelo.predict(teste_X_norm)
media = np.sqrt(mean_absolute_error(teste_y, previsao))
print("Média de erro:", media)
prec = modelo.score(teste_X_norm, teste_y)
print("Precisão: ",prec*100)
#---------- treinando modelo -------------#

n = previsao.size
treino = treino_y#.iloc[:tam_treino]
validacao = y.iloc[tam_treino:]
validacao['Previsto'] = previsao


def graf_1():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=treino.index,
        y=treino['Futuro'],
        name='Treino'
    ))
    fig1.add_trace(go.Scatter(
        x=validacao.index,
        y=validacao['Futuro'].iloc[:n-1],
        name='Teste Real'
    ))
    fig1.add_trace(go.Scatter(
        x=validacao.index,
        y=validacao['Previsto'],
        name='Teste Previsto'
    ))
    fig1.update_layout(
        title={
        'text': "<b>Gráfico do modelo</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title='Real(R$)',
        xaxis_title='Período',
        legend_title='Labels',
        width=800
    )
    st.write(fig1)
    arq3 = open('grafico2.txt','r',encoding='utf-8')
    t3 = arq3.readline()
    st.write(t3)

#---------- Previsão do dia -------------#
with st.sidebar:
    st.sidebar.title('Menu')
    opcoes =  st.sidebar.selectbox('Selecione', [f'Sobre {sigla}', 'Sobre o modelo'])

if opcoes==f"Sobre {sigla}":
    if st.button('Buscar'):
        mostra_dados()
        graf_2()

elif opcoes=="Sobre o modelo":
    graf_1()
    st.subheader('Gerar previsão')
    # pega o último valor de features para ser usado na previsão
    n = teste_X.shape[0]
    amanha = teste_X_norm[n-1:]
    dia_seguinte = modelo.predict(amanha)
    dia_seguinte = str(*dia_seguinte)
    if st.button('Gerar'):
        prec = '{:.2f}'.format(prec*100)
        st.write(f'A precisão do modelo na ação {sigla} é de {prec}%')
        st.write(f"Previsão do preço de fechamento para o próximo dia: R$ {dia_seguinte}")





   


