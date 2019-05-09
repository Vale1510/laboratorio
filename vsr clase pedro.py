# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:41:50 2018

@author: Martin
"""

import numpy as np
import pandas as pd
import seaborn.apionly as sns
import matplotlib.pyplot as plt


from sklearn.svm import SVR


home_banking = pd.read_excel(r'E:\Univ. Austral\Ferola Laboratorio\Libro4.xlsx')



from sklearn.metrics import explained_variance_score
    


#%%



## leo las 2 hojas del excel con los datos de visistas y vencimiento de tarjetas

home_banking = pd.read_excel(r'E:\Univ. Austral\Ferola Laboratorio\home_bamking_may2018.xlsx',sheet_name ='Data set Raw')
tarjetas=pd.read_excel(r'E:\Univ. Austral\Ferola Laboratorio\home_bamking_may2018.xlsx',sheet_name ='Venc Tarjeas Visa y Master')
hb=home_banking.copy()

hb=hb.dropna()

hb['fecha']=pd.to_datetime(hb["fecha"],format='%Y/%m/%d')
hb['visa']=pd.to_datetime(hb["visa"],format='%Y/%m/%d')
hb['master']=pd.to_datetime(hb["master"],format='%Y/%m/%d')




# vector de fechas true/false vencimiento de atrjetas

visa=hb['visa']!='1970-01-01 00:00:00'
master=hb['master']!='1970-01-01 00:00:00'
coinciden=hb['coinciden']
lunes_NO_venc_visa=[0 if i else 1 for i in visa]
lunes_NO_venc_master=[0 if i else 1 for i in master]

        
## busco el ciclo mas parecido
hb['lag_vencimiento_homologo_visa']=0
hb['indice_semana_venc_lag_visa']=0

        
        
real_=np.array(hb['real'])
hb.columns

## Calculo indice donde empieza visa y mastercard en el archivo de vencimientos
ind_visa_min=tarjetas[tarjetas['Tarjetas']=='TV'].index.min()
ind_master_min=tarjetas[tarjetas['Tarjetas']=='TM'].index.min()

## PRIMER VARIABLE CALCULADA: lag_vencimiento_homologo aplica a Visasiempre y a mastercard cuando coinciden los venc con visa
c=0
for i,real in enumerate(hb['real']):
    #i=580
    if visa[i]:# entra solo si ese dia hay un vecimiento de visa
        c+=1 # cuento el orden de los vecnimientos de la serie
        
        # busco el indice del venciumineto en tarjetas a partir de la fecha de vencimiento
        # de hb
        xx=hb.loc[i,'visa']
        valor=min(list(tarjetas.loc[ind_visa_min:ind_visa_min+c,'Vence']), key=lambda x:abs(x-xx))
        ind_valor_visa=list(tarjetas.loc[ind_visa_min:,'Vence']).index(valor)+ind_visa_min #ind tarjetas
        
        if ind_valor_visa >=ind_visa_min+4:
            # calculo vencimiento analogo anterior
            xx=tarjetas.loc[ind_valor_visa-4,'Vence']
            if i <36:
                menos=0
                de=0
            else:
                menos=35
                de=i
                
                """""el i =34 no lo encuentra..revizar"""""
                
            valor=min(hb[de-menos:i]['visa'], key=lambda x:abs(x-xx))
            ind_valor_visa_hb_lag=list(hb.loc[de-menos:i,'visa']).index(valor)+de-menos
            hb.loc[ind_valor_visa_hb_lag,'visa']
            
            # chequeo si el dia de vencimiento es feriado..imputo el primer dia habil
            i_temp_lag=ind_valor_visa_hb_lag
            if hb.loc[ind_valor_visa_hb_lag,'feriado']==1:
                
                while hb.loc[i_temp_lag,'habil']==0:
                    i_temp_lag+=1
                    
            # cheuqe si el dia actual i es feriado e imputo el primer dia habil
            i_temp=i
            if hb.loc[i,'feriado']==1:
                
                while hb.loc[i_temp,'habil']==0:
                    i_temp+=1
          
            
            hb.loc[i_temp,'lag_vencimiento_homologo_visa']=hb.loc[i_temp_lag,'real']
            corte=5-hb.loc[i_temp,'Tipo de dia']
            if (i_temp+corte)>len(hb)-1:
                corte=len(hb)-1-i_temp
            hb.loc[i_temp:(i_temp+corte),'indice_semana_venc_lag_visa']=i_temp_lag
            
        else:
            xx=hb.loc[i,'real']
            valor=min(hb[:i]['real'], key=lambda x:abs(x-xx))
            ind_valor=list(real_[:i]).index(valor)
             # chequeo si el dia de vencimiento es feriado..imputo el primer dia habil
            
           
            hb.loc[i,'lag_vencimiento_homologo_visa']=hb.loc[ind_valor,'real']
            
 ### mastercard #####################################################


hb['indice_semana_venc_lag_master']=0
hb['lag_vencimiento_homologo_master']=0

c=0
for i,real in enumerate(hb['real']):
   
    if master[i]:# entra solo si ese dia hay un vecimiento de visa
        c+=1 # cuento el orden de los vecnimientos de la serie
        
        # busco el indice del venciumineto en tarjetas a partir de la fecha de vencimiento
        # de hb
        xx=hb.loc[i,'master']
        valor=min(list(tarjetas.loc[ind_master_min:ind_master_min+c,'Vence']), key=lambda x:abs(x-xx))
        ind_master_lag=list(tarjetas.loc[ind_master_min:,'Vence']).index(valor)+ind_master_min #ind tarjetas
        
        if ind_master_lag >=ind_master_min+2:
            # calculo vencimiento analogo anterior
            d1=(tarjetas.loc[ind_master_lag,'Vence']-tarjetas.loc[ind_master_lag-1,'Vence']).days
            d2=(tarjetas.loc[ind_master_lag,'Vence']-tarjetas.loc[ind_master_lag-2,'Vence']).days
            if d1==28:
                xx=tarjetas.loc[ind_master_lag-1,'Vence']
            elif d1<28 and (d2==28 or d2==35):
                xx=tarjetas.loc[ind_master_lag-2,'Vence']
          
            if i <36:
                menos=0
                de=0
            else:
                menos=35
                de=i
                
                """""el i =34 no lo encuentra..revizar"""""
                
            valor=min(hb[menos:i]['master'], key=lambda x:abs(x-xx))
            ind_master_min_hb_lag=list(hb.loc[de-menos:i,'master']).index(valor)+de-menos
            hb.loc[ind_master_min_hb_lag,'real']
            
            # chequeo si el dia de vencimiento es feriado..imputo el primer dia habil
            i_temp_lag=ind_master_min_hb_lag
            if hb.loc[ind_master_min_hb_lag,'feriado']==1:
                
                while hb.loc[i_temp_lag,'habil']==0:
                    i_temp_lag+=1
                    
            # cheuqe si el dia actual i es feriado e imputo el primer dia habil
            i_temp=i
            if hb.loc[i,'feriado']==1:
                
                while hb.loc[i_temp,'habil']==0:
                    i_temp+=1
          
            
            hb.loc[i_temp,'lag_vencimiento_homologo_master']=hb.loc[i_temp_lag,'real']
            corte=5-hb.loc[i_temp,'Tipo de dia']
            if (i_temp+corte)>len(hb)-1:
                corte=len(hb)-1-i_temp
            hb.loc[i_temp:(i_temp+corte),'indice_semana_venc_lag_master']=i_temp_lag
            
            
        else:
            xx=hb.loc[i,'real']
            valor=min(hb[:i]['real'], key=lambda x:abs(x-xx))
            ind_valor=list(real_[:i]).index(valor)
             # chequeo si el dia de vencimiento es feriado..imputo el primer dia habil
            
           
            hb.loc[i,'lag_vencimiento_homologo_master']=hb.loc[ind_valor,'real']  
            
            
            
## promedios de feriados y posteferiados por tipo de dia
Q_feriado=hb.groupby(['Tipo de dia','feriado']).real.mean()
Q_post_fer=hb.groupby(['Tipo de dia','posferiado']).real.mean()
Q_post_habil=hb.groupby(['mm','Tipo de dia','habil']).real.mean()

hb['lunes_NO_ven_Master']=lunes_NO_venc_master
hb['lunes_NO_ven_Visa']=lunes_NO_venc_visa
hb['Lunes_NO_Venc'] = [a*b for a,b in zip(lunes_NO_venc_visa, lunes_NO_venc_visa)]

Q_post_Lunes_No_Venci=hb[hb['Tipo de dia']==1].groupby(['mm','semana mes','Lunes_NO_Venc']).real.mean()


Q_post_habil=hb.groupby(['mm','Tipo de dia','habil']).real.mean()


# aacumula por mes de real
Q_acum_real=hb.groupby(['yy','mm']).real.sum()

Q_acum_real[1][11]
# uso nov del priemr año como base para normalizar

max_index_base=Q_acum_real[1][11]
## unificación de vencimientos master y visa..me quedo con el maximo
hb['lag_venc']=0
hb['lag_7']=0
hb['lag_1']=0
hb['lag_t-Est']=0
hb['lag_t-Homologo']=0


hb.columns
acu_=0
acu=0
venc_lunes=9999
indice_lag=0
for i,real in enumerate(hb['real']):
    
    
    if i>9:
    #i=595
    #hb.loc[i,'real']
        if (hb.loc[i,'lag_vencimiento_homologo_master']>0 and hb.loc[i,'lag_vencimiento_homologo_visa']>0):
            hb.loc[i,'lag_venc']=max(hb.loc[i,'lag_vencimiento_homologo_master'],hb.loc[i,'lag_vencimiento_homologo_visa'])
            venc_lunes=hb.loc[i,'lag_venc']
                
        elif hb.loc[i,'lag_vencimiento_homologo_master']>0:
            hb.loc[i,'lag_venc']=hb.loc[i,'lag_vencimiento_homologo_master']
            venc_lunes=hb.loc[i,'lag_venc']
        elif hb.loc[i,'lag_vencimiento_homologo_visa']>0:
            hb.loc[i,'lag_venc']=hb.loc[i,'lag_vencimiento_homologo_visa']
            venc_lunes=hb.loc[i,'lag_venc']
        # dias distintoal vencimiento   
        
        if hb.loc[i,'lag_venc']==0:
                # Fin de semana
            im=hb.loc[i,'indice_semana_venc_lag_master']
            iv=hb.loc[i,'indice_semana_venc_lag_visa']
            
            if hb.loc[im,'real']==venc_lunes and im!=0:
                indice_lag=hb.loc[i,'indice_semana_venc_lag_master']
            elif hb.loc[iv,'real']==venc_lunes and iv!=0:
                indice_lag=hb.loc[i,'indice_semana_venc_lag_visa']
            else:
                pass
            
            if hb.loc[i,'Tipo de dia'] in [1]:
                
                mes=hb.loc[i,'mm']
                semana=hb.loc[i,'semana mes']
                if hb.loc[i,'feriado']==1:
                    hb.loc[i,'lag_1']=Q_feriado[hb.loc[i,'Tipo de dia']][1] 
                else:
                    hb.loc[i,'lag_7']=Q_post_Lunes_No_Venci[mes][semana][1]
            if hb.loc[i,'Tipo de dia'] in [6,7]:
                hb.loc[i,'lag_7']=hb.loc[i-7,'real']
            if hb.loc[i,'Tipo de dia'] in [4,5]:
                #- jueves y viernes normales actuales y de la ultima semana
                if hb.loc[i,'feriado']==0 and hb.loc[i,'posferiado']==0 and hb.loc[i-7,'feriado']==0 and hb.loc[i-7,'posferiado']==0:
                
                    hb.loc[i,'lag_1']=hb.loc[i-1,'real']
                    hb.loc[i,'lag_7']=hb.loc[i-7,'real']
                    tdd=hb.loc[indice_lag,'Tipo de dia']
                    tddi=hb.loc[i,'Tipo de dia']
                    lag_pos=tddi-tdd+indice_lag
                    hb.loc[i,'lag_t-Homologo']=hb.loc[lag_pos,'real']
                    ## REVIZAR EL HOMOLOGO DE LOS DIAS NORMALES
                    # PENSAR SI TOMO EL MAXIMO DE LAS ALETRNTIVAS
                    
                    if hb.loc[i-8,'posferiado']==1:# si i=jueves  i-8=miercoles i-9=martes (Solo i=miercols=3)
                        # media de dia t-8 habil normal del mes
                        dias_8=Q_post_habil.loc[hb.loc[i-8,'mm']][hb.loc[i-8,'Tipo de dia']][1]
                        # implica que i-9=lunes es feriado
                        dias_9=Q_post_habil.loc[hb.loc[i-9,'mm']][hb.loc[i-9,'Tipo de dia']][1]
                    else:
                        dias_8=hb.loc[i-8,'real']
                        dias_9=hb.loc[i-9,'real']
                        
                       
                            
                    if hb.loc[i-1,'posferiado']==1:
                        
                        dias_1=Q_post_habil.loc[hb.loc[i-1,'mm']][hb.loc[i-1,'Tipo de dia']][1]
                        dias_2=Q_post_habil.loc[hb.loc[i-2,'mm']][hb.loc[i-2,'Tipo de dia']][1]
                    else:
                        dias_1=hb.loc[i-1,'real']
                        dias_2=hb.loc[i-2,'real']
                    
                    temp=dias_2*dias_8/dias_9
                    tasa=dias_1/temp
                    tasa_t=dias_1*hb.loc[i-7,'real']/dias_8
                    tasa_recal=tasa_t/dias_1
                    tasa_ajus=tasa_recal*0.2+0.8*tasa
                    
                    
                    if hb.loc[i,'habil']==1:
                        hb.loc[i,'lag_t-Est']=dias_1*tasa_ajus
                    elif hb.loc[i-1,'habil']==1:
                        hb.loc[i,'lag_t-Est']=Q_post_habil.loc[hb.loc[i-1,'mm']][hb.loc[i-1,'Tipo de dia']][1]*hb.loc[i-7,'real']/dias_8
                    else:
                        
                        if tasa<=1:
                            hb.loc[i,'lag_t-Est']=tasa_t*tasa
                        else:
                            hb.loc[i,'lag_t-Est']=tasa_t*tasa
                    
                    
                    #__________________________________________________________
                    '''
                    if hb.loc[i-8,'feriado']==0 and hb.loc[i-8,'posferiado']==0:
                        
                        temp=hb.loc[i-2,'real']*hb.loc[i-8,'real']/hb.loc[i-9,'real']
                        tasa=hb.loc[i-1,'real']/temp
                        tasa_t=hb.loc[i-1,'real']*hb.loc[i-7,'real']/hb.loc[i-8,'real']
                        if temp<=1:
                            hb.loc[i,'lag_t-Est']=tasa_t*tasa
                        else:
                            hb.loc[i,'lag_t-Est']=tasa_t*tasa
                     '''       
                    #__________________________________________________________
                    
                # jueves y viernes feriado o postferiado..le imputo el promedio
                if hb.loc[i,'feriado']==1:
                    hb.loc[i,'lag_1']=Q_feriado[hb.loc[i,'Tipo de dia']][1]
                   
                        
                if hb.loc[i,'posferiado']==1:
                    try:
                        
                        hb.loc[i,'lag_1']=Q_post_fer[hb.loc[i,'Tipo de dia']][1]
                    except:
                            
                        hb.loc[i,'lag_1']=Q_post_fer[5][1]
                    
            if hb.loc[i,'Tipo de dia'] in [2,3]:
                    #- martes  y miercoles normales actuales y de la ultima semana
                    
                    if hb.loc[i,'feriado']==0 and hb.loc[i,'posferiado']==0 and hb.loc[i-7,'feriado']==0 and hb.loc[i-7,'posferiado']==0:
                        
                        hb.loc[i,'lag_1']=hb.loc[i-1,'real']
                        hb.loc[i,'lag_7']=hb.loc[i-7,'real']
                        
                        hb.loc[i,'lag_1']=hb.loc[i-1,'real']
                        hb.loc[i,'lag_7']=hb.loc[i-7,'real']
                        
                        tdd=hb.loc[indice_lag,'Tipo de dia']
                        tddi=hb.loc[i,'Tipo de dia']
                        lag_pos=tddi-tdd+indice_lag
                        hb.loc[i,'lag_t-Homologo']=hb.loc[lag_pos,'real']
                        
                        if hb.loc[i-8,'posferiado']==1:# si i=miercoles  i-8=martes i-9=lunes (Solo i=miercols=3)
                            # media de dia t-8 habil normal del mes
                            dias_8=Q_post_habil.loc[hb.loc[i-8,'mm']][hb.loc[i-8,'Tipo de dia']][1]
                            # implica que i-9=lunes es feriado
                            dias_9=Q_post_habil.loc[hb.loc[i-9,'mm']][hb.loc[i-9,'Tipo de dia']][1]
                        else:
                            dias_8=hb.loc[i-8,'real']
                            dias_9=hb.loc[i-9,'real']
                        
                       
                            
                        if hb.loc[i-1,'posferiado']==1:
                            
                            dias_1=Q_post_habil.loc[hb.loc[i-1,'mm']][hb.loc[i-1,'Tipo de dia']][1]
                            dias_2=Q_post_habil.loc[hb.loc[i-2,'mm']][hb.loc[i-2,'Tipo de dia']][1]
                        else:
                            dias_1=hb.loc[i-1,'real']
                            dias_2=hb.loc[i-2,'real']
                            
                            
                        temp=dias_2*dias_8/dias_9
                        tasa=dias_1/temp
                        tasa_t=dias_1*hb.loc[i-7,'real']/dias_8
                        
                        tasa_recal=tasa_t/dias_1
                        tasa_ajus=tasa_recal*0.2+0.8*tasa
                        
                        
                        if hb.loc[i,'habil']==1:
                            hb.loc[i,'lag_t-Est']=dias_1*tasa_ajus
                        #elif hb.loc[i-1,'ultimo_habil']==1:
                            #hb.loc[i,'lag_t-Est']=Q_post_habil.loc[hb.loc[i-1,'mm']][hb.loc[i-1,'Tipo de dia']][1]*hb.loc[i-7,'real']/dias_8
                        else:
                        
                            if tasa<=1:
                                hb.loc[i,'lag_t-Est']=tasa_t*tasa
                            else:
                                hb.loc[i,'lag_t-Est']=tasa_t*tasa
                            
                            
                            
                            
                            
                    # martes y miercoles feriado o postferiado..le imputo el promedio
                    if hb.loc[i,'feriado']==1:
                        
                        try:
                            hb.loc[i,'lag_1']=Q_feriado[hb.loc[i,'Tipo de dia']][1]
                        except:
                            # el problema esta en el tipo de dia 3 (miercoels)...imputo jueves 4
                            hb.loc[i,'lag_1']=Q_feriado[2][1]
                        
                    if hb.loc[i,'posferiado']==1:
                        hb.loc[i,'lag_1']=Q_post_fer[hb.loc[i,'Tipo de dia']][1]
            
                    
                
        if hb.loc[i,'dd']==1 :
            
            acu_=0
            if hb.loc[i,'feriado']==1:
                acu=Q_feriado[hb.loc[i,'Tipo de dia']][1]/max_index_base
            if hb.loc[i,'posferiado']==1:
                acu=Q_post_fer[hb.loc[i,'Tipo de dia']][1]/max_index_base
            else:
                acu=Q_post_habil.loc[hb.loc[i,'mm']][hb.loc[i,'Tipo de dia']][hb.loc[i,'habil']]/max_index_base
            hb.loc[i,'lag_Acum_mes']=acu
        else:
            
            if  hb.loc[i,'dd']==2 :
                
                acu_=acu_+acu
            
            else:
                
                acu_=acu_+hb.loc[i-1,'real']/max_index_base
                
            hb.loc[i,'lag_Acum_mes']=acu_   
       
 #%% data set elimine despues de ver resultados
       'pred','semana','Largo del mes','Preferiado','lag_Acum_mes','lag_1','lag_t-Est',
       max2=X['pred'].max()
min2=X['pred'].min()

X['pred']=[52*(i-min2)/(max2-min2) for i in X['pred']]
#%% Data set 
hb.columns        
X1=hb[[ 'real', 'Tipo de dia', 'semana mes','Semana Normal', 'semana corta 4', 'Semana corta3',\
       'coinciden', 'yy', 'mm', \
        'finde', 'feriado', 'posferiado', 'habil', 'primer_habil','ultimo_habil', \
        'lag_venc', 'lag_7',  'lag_t-Homologo']]





fecha_=hb[10:].fecha
X=X1[10:].drop('real',axis=1)     
y=X1[10:].real
X.columns
X.reset_index(drop=True)
y.reset_index(drop=True)



maxy=y[:590].max()
miny=y[:590].min()


y=[52*(i-miny)/(maxy-miny) for i in y]
y=pd.DataFrame(y,index=X.index)

X['lag_venc']=[52*(i-miny)/(maxy-miny) if i>0 else 0 for i in X['lag_venc']]

X['lag_7']=[52*(i-miny)/(maxy-miny) if i>0 else 0 for i in X['lag_7']]

X['lag_1']=[52*(i-miny)/(maxy-miny) if i>0 else 0 for i in  X['lag_1']]
X['lag_t-Est']=[52*(i-miny)/(maxy-miny) if i>0 else 0 for i in  X['lag_t-Est']]
X['lag_Acum_mes']=[52*i if i>0 else 0 for i in  X['lag_Acum_mes']]
X['lag_t-Homologo']=[52*(i-miny)/(maxy-miny) if i>0 else 0 for i in  X['lag_t-Homologo']]



largo=int(len(X) *(606-31)/606)
largo=594

X_train=X.loc[:largo] 
y_train=y.loc[:largo]           
X_valid=X.loc[largo+1:603]
y_valid=y.loc[largo+1:603]
type(y)

## VECTOR SUPPORT MACHINE
#C=2000, gamma=0.00085959


lambdas = np.linspace(2040, 3000, 10)
gammas = np.linspace(0.0030,  0.00035, 20)

lambdas = np.linspace(7500, 15000, 10)
gammas = np.linspace(0.0009384,  0.002, 40)

'lag_t-Homologo'
param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])

res = pd.DataFrame([], index=[str(d) for d in param_grid], columns=['parametros',"train", "valid"])
max_var_exp=-9999999
def iteracion(params):
    global max_var_exp,max_var_train,c_opt,sigma_op
    
    C =  params[0]
    #print(params[0],params[1])
    sigma =  params[1]
    
    learner=SVR(C=C, gamma=sigma,epsilon=0.5,kernel='rbf')
    
    
    learner.fit(X_train, y_train)
    max_var_exp_=explained_variance_score(y_valid, learner.predict(X_valid))
    max_var_train_=explained_variance_score(y_train, learner.predict(X_train))
    #print(max_var_exp_)
    res.loc[str(params), "train"] = max_var_train_
    res.loc[str(params), "valid"] = max_var_exp_
    if max_var_exp_>max_var_exp:
        max_var_exp=max_var_exp_
        max_var_train=max_var_train_
        c_opt=C
        sigma_op=sigma
  
    print(params[0],params[1],'  ',"{:.7f}".format(res.loc[str(params), "valid"]))
    return res


real_loss = [iteracion(params) for params in param_grid]

152.63157
0.002586
3.5

100
0.003
epsilon 2.5
#%%





# tomamos el producto cartesiano entre los dos arrays de parametros


    



# tomamos el producto cartesiano entre los dos arrays de parametros
param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])






### RANDOM FOREST REGRESION

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import explained_variance_score


depths = list(range(2, 40, 2)) + [None]


# predict varianza
res = pd.DataFrame([], index=[str(d) for d in depths], columns=["train", "valid"])

for depth in depths:
    
    learner = RandomForestRegressor(n_estimators=150, max_depth=depth)
    learner.fit(X_train, y_train)
    res.loc[str(depth), "train"] = explained_variance_score(y_train, learner.predict(X_train))
    res.loc[str(depth), "valid"] = explained_variance_score(y_valid, learner.predict(X_valid))
    learner.feature_importances_


import seaborn.apionly as sns
sns.set()          
Bbox([[0.125, 0.10999999999999999], [0.9, 0.88]])
ax = plt.subplot(111)
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 , .12 ,  0.9 , 0.88] 
plot.set_position(pos2) # set a new position
    

    
    
## grafico del calculo de metaparametros
y.min()
ax = res.plot.line(grid=True, fontsize=30,ylim=(0, 1), linestyle='-', style='o')
ax.set_xticks(range(len(res)))
ax.set_xticklabels(res.index)
ax.legend(fontsize=30)         
            
#%%
#### Graficas para generar nuevas variables
            

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def system(xs,ys,zs,color,m, ax):

        # ...
        
        ax.scatter(xs, ys, zs,c=color,marker=m, linewidth=2,s=100)

xs=0
ys=0
zs=0
n = 100
color=['r','b','g','y']
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
ppp=0
data=np.zeros((40,3),dtype=float)



ccc=0
fig3d=plt.figure()
ax=Axes3D(fig3d)

semana_del_mes=[1,2,3,4,5]
for ii in semana_del_mes:
    
    ccc=ii
       
    ppp+=1
  
    aa=np.array(hb['real'][(hb['semana mes'] == ii) & (hb['feriado'] == 0) & (hb['posferiado'] == 0)])
    
    bb=np.array(hb['Tipo de dia'][(hb['semana mes']==ii) & (hb['feriado'] == 0) & (hb['posferiado'] == 0)])
    real=list(aa)
    tipo_dia=list(bb)
    
    real = [float(i) for i in real]
    tipo_dia = [float(i) for i in tipo_dia]
    xs = list(np.repeat(ii, len(real)))
    
    ys =tipo_dia
    zs = real
    if ppp==3:
        ppp=0
    system(xs,ys,zs,color[ppp],'.', ax)
    

plt.show()
   
    
ax.set_xlabel('Tipo de semana')
ax.set_ylabel('Tipo de Dia')
ax.set_zlabel('Visitas')
ax.xaxis.labelpad = 40
ax.yaxis.labelpad = 40
ax.zaxis.labelpad = 40

plt.show()  

#%%
X.columns
xx_pred=[i for i in range(len(y_pred))]
y_pred=hb[hb['lag_t-Est']>0]['lag_t-Est']
y_pred=[int(i) for i in y_pred]

y_real=hb[hb['lag_t-Est']>0]['real']
c=0
for i in y_pred.index:
    if 0.8 >y_pred.loc[i]/y_real.loc[i] or y_pred.loc[i]/y_real.loc[i]>1.2:
        print(i,'  ', y_pred.loc[i],'  ',y_real.loc[i] )
        print(hb.loc[i,'Tipo de dia'])
        c+=1
y_real=hb[(hb['Tipo de dia']==1) & (hb['semana mes']==2)]['real']
xx_pred=hb[(hb['Tipo de dia']==1) & (hb['semana mes']==2)]['lag_Acum_mes']
plt.scatter(xx_pred, y_pred,  color='red', alpha=0.5,marker='*',s=tt*2)#,s=tt)
plt.plot(xx_pred, y_real, color='b', alpha=0.5)#,s=tt)
plt.show()
#grfico aislados de variables

##############################

X_train=X[:largo] 
y_train=y[:largo]           
X_valid=X[largo:]
y_valid=y[largo:]

y_est_train=learner.predict(X_train)
y_est_valid=learner.predict(X_valid)

X_valid=X.loc[largo+1:]

len(X_valid)


##################
X_train=X[:largo] 
y_train=y[:largo]           
X_valid=X[largo:]
y_valid=y[largo:]

y_est_train=learner.predict(X_train)
y_est_valid=learner.predict(X_valid)
x_val_g=[i for i in range(largo,largo+len(y_est_valid))]
x_train_g=[i for i in range(largo)]
len(y_train)


y=[52*(i-miny)/(maxy-miny) for i in y]

y_train=[i*(maxy-miny)/52+miny for i in y_train ]
y_valid=[i*(maxy-miny)/52+miny for i in y_valid ]
y_est_train=[i*(maxy-miny)/52+miny for i in y_est_train ]
y_est_valid=[i*(maxy-miny)/52+miny for i in y_est_valid ]

indi=X_valid.index
y_viejo=hb.loc[indi,'pred']
hb.columns

bg_color = 'black'
fg_color = 'white'
fig = plt.figure(1,figsize=(25,5),facecolor=bg_color, edgecolor=fg_color)
plt.figure(1,figsize=(25,5),axisbg='red')
axes = plt.axes((0.1, 0.1, 0.8, .8), axisbg=bg_color)
axes.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
axes.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
axes.axhline(y=0, color='w')
axes.axvline(x=0, color='w')  
axes.grid(True, which='both')

for spine in axes.spines.values():
    spine.set_color(fg_color)

plt.text(-0.48, 0.35, 'gamma = 15', fontsize=12)
plt.title('Serie de tiempo Visitas Homebanking')
plt.xlabel('Tiempo')
plt.ylabel('Visitas')


plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 300
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
#plt.figure(figsize=(25,6))





tt=250
tt0=500
tt1=500
tt2=500




len(y_est_valid)
plt.plot(x_train_g, y_train, color='red', alpha=0.5)#,s=tt)


plt.plot(x_train_g ,y_est_train, color='y',marker='*', alpha=0.5)#,s=tt)

plt.plot(x_val_g, y_valid, color='w', alpha=0.5) #,s=tt*1)
plt.scatter(x_val_g, y_est_valid, color='orange', alpha=0.5,marker='*',s=tt*2)

#plt.scatter(x_val_g, y_viejo, color='y', alpha=0.5,marker='o',s=tt*3)


plt.title('Home banking')
plt.ylabel('Visitas')
plt.xlabel('Tiempo')

plt.show()

          
            
 #%%           
            
           
bg_color = 'black'
fg_color = 'white'
fig = plt.figure(1,figsize=(25,5),facecolor=bg_color, edgecolor=fg_color)
plt.figure(1,figsize=(25,5),axisbg='red')
axes = plt.axes((0.1, 0.1, 0.8, .8), axisbg=bg_color)
axes.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
axes.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
axes.axhline(y=0, color='w')
axes.axvline(x=0, color='w')  
axes.grid(True, which='both')

for spine in axes.spines.values():
    spine.set_color(fg_color)

plt.text(-0.48, 0.35, 'gamma = 15', fontsize=12)
plt.title('Serie de tiempo Visitas Homebanking')
plt.xlabel('Tiempo')
plt.ylabel('Visitas')
hb['Prod_feria_postfer']=hb['feriado']+hb['posferiado']

Xc, y = hb['dias acum'][hb['Prod_feria_postfer'] == 0], hb['real'][hb['Prod_feria_postfer'] == 0 ]
y1=hb[['semana mes','Tipo de dia']][hb['Prod_feria_postfer'] == 0]
len(y1)
len(y)
len(Xc)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 300
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
#plt.figure(figsize=(25,6))
dia=1
select_indices1 = ((y1['semana mes']==1) & (y1['Tipo de dia']==dia))
select_indices1=select_indices1[select_indices1==True].index
select_indices2 = ((y1['semana mes']==2) & (y1['Tipo de dia']==dia))
select_indices2=select_indices2[select_indices2==True].index
select_indices3 = ((y1['semana mes']==3) & (y1['Tipo de dia']==dia))
select_indices3=select_indices3[select_indices3==True].index
select_indices4 = ((y1['semana mes']==4) & (y1['Tipo de dia']==dia))
select_indices4=select_indices4[select_indices4==True].index
select_indices5 = ((y1['semana mes']==5) & (y1['Tipo de dia']==dia))
select_indices5=select_indices5[select_indices5==True].index


hb.columns
'''
select_indices1 = np.where( np.logical_and( y1 <30,y1.index<860 ))
select_indices11 = np.where( np.logical_and( y1 <30,y1.index>859 ))
select_indices2 = np.where( np.logical_and(  np.logical_and(y1 >=29,y<200),y1.index<860 ))
select_indices22 = np.where( np.logical_and( np.logical_and(y1 >=29,y<200),y1.index>859 ))
select_indices3 = np.where( np.logical_and( y1 >199,y1.index<860 ))
select_indices33 = np.where( np.logical_and( y >199,y.index>859 ))


select_indices44=y[(y['POZO']=='PO-1055') & (y['dias_fallas_mas_cercana']<60)].index
select_indices45=y[(y['POZO']=='PO-1055') & (y['dias_fallas_mas_cercana']>59)].index
'''    

type(Xc)
tt=250
tt0=500
tt1=500
tt2=500
plt.scatter(Xc.loc[select_indices1].dropna(), y.loc[select_indices1].dropna(), color='red', alpha=0.5,s=tt)
plt.scatter(Xc.loc[select_indices2].dropna(), y.loc[select_indices2].dropna(), color='b', alpha=0.5,s=tt)
plt.scatter(Xc.loc[select_indices3].dropna(), y.loc[select_indices3].dropna(), color='y', alpha=0.5,s=tt)

plt.scatter(Xc.loc[select_indices4].dropna(), y.loc[select_indices4].dropna(), color='mg', marker='*', alpha=0.5,s=tt*3)
plt.scatter(Xc.loc[select_indices5].dropna(), y.loc[select_indices5].dropna(), color='w', marker='*',alpha=0.5,s=tt*3)


plt.title('Cerro Dragon - 25 pozos + 58 pozos Zorro + Valle Hermoso')
plt.ylabel('Coordenada CP I')
plt.xlabel('Coordenada CP II')

plt.show() 
            
            
            