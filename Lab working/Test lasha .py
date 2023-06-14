import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

l = 3               # m length of the cubic room
Sg = l**2           # m² surface of the glass wall
Sc = Si = 5 * Sg    # m² surface of concrete & insulation of the 5 walls
n=10


air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
# pd.DataFrame.from_dict(air, orient='index', columns=['air'])
pd.DataFrame(air, index=['Air'])



#tous les 10 au niveau des surfaces sont a supprimer, je voulais juste voir si ca marchait 

concrete1_2 = {'Conductivity': 1.400,
            'Density': 2300.0,
            'Specific heat': 880,
            'Width': 0.25,
            'Surface':16.3}   #wall1,wall2

concrete3 = {'Conductivity': 1.400,
            'Density': 2300.0,
            'Specific heat': 880,
            'Width': 0.25,
            'Surface':17.75}   #wallindoor

insulation1_2 = {'Conductivity': 0.027,
              'Density': 55.0,
              'Specific heat': 1210,
              'Width': 0.05,
              'Surface':15.7 }     #wall1,wall2


insulation3 = {'Conductivity': 0.027,
              'Density': 55.0,
              'Specific heat': 1210,
              'Width': 0.05,
              'Surface':  3}     #wallindoor

glass = {'Conductivity': 1.4,
         'Density': 2500,
         'Specific heat': 1210,
         'Width': 0.04,
         'Surface': 3}     #window1,window2

wood1 = {'Conductivity': 0.2,
         'Density': 2500,
         'Specific heat': 720,
         'Width': 0.05,
         'Surface':10}     #door1

wood2 = {'Conductivity': 0.2,
         'Density': 2500,
         'Specific heat': 720,
         'Width': 0.05,
         'Surface':10 }     #door2

wall = pd.DataFrame.from_dict({'wall1_2' :concrete1_2, 'indwall' : concrete3, 'layerout1_2' : insulation1_2, 'layerout3' : insulation3, 'glass' : glass,
                              'door1' : wood1, 'inddoor' : wood2},
                              orient='index')

"""
controller1 = pd.DataFrame.from_dict({},
                              orient='index')

controller2 = pd.DataFrame.from_dict({},
                              orient='index')

ventilation = pd.DataFrame.from_dict({},
                              orient='index')
"""

wall






# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass





σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant
print(f'σ = {σ} W/(m²⋅K⁴)')




h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
# h = pd.DataFrame([{'in': 8., 'out': 25}])  # W/(m²⋅K)
h






# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
pd.DataFrame(G_cd, columns=['Conduction'])





# convection
Gcv_in = h['in'][0]* wall['Surface']    
Gcv_out = h['out'][0] * wall['Surface']     

Gcv = { 'Gcv_in' : Gcv_in, 'Gcv_out' : Gcv_out}
df = pd.DataFrame.from_dict(Gcv, orient='columns')
print(df)

print(Gcv_out['wall1_2'])









# view factor wall-glass
Fwg1_2 = glass['Surface'] / concrete1_2['Surface']
Fwg3=glass['Surface'] / concrete3['Surface']

Fwg = { 'Fwg1_2' : Fwg1_2, 'Fwg3' : Fwg3}
df = pd.DataFrame.from_dict(Fwg, orient='index')
print(df)








# long wave radiation
Tm = 20 + 273   # K, mean temp for radiative exchange
#wall1_2
GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['wall1_2']
GLW12 = 4 * σ * Tm**3 * Fwg1_2 * wall['Surface']['wall1_2']
GLWG = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['glass']

#indwall
GLWIND = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['indwall']
GLWINDG = 4 * σ * Tm**3 * Fwg3 * wall['Surface']['indwall']









GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLWG)
GLWW = 1 / (1 / GLWIND + 1 / GLWINDG + 1 / GLWG)

print('long wave radiation wall1_2/glass =', GLW, 'W/K')
print('long wave radiation wall3/glass =', GLWW, 'W/K')








# ventilation flow rate
Va = 2*638*742.5*2.5                 # m³, volume of air, two rooms * L * l * h
ACH = 1                              # air changes per hour
Va_dot = (ACH / 3600) * Va           # m³/s, air infiltration
print('Va_dot=',Va_dot,'m^3')







# ventilation & advection
Gv = air['Density'] * air['Specific heat'] * Va_dot
Gv 









# P-controler gain
Kp = 1e4            # almost perfect controller Kp -> ∞
#Kp = 1e-3           # no controller Kp -> 0
#Kp = 0








# glass: convection outdoor & conduction
#Ggs = float(1 / (1 /Gcv_out['out'] + 1 / (2 * G_cd['glass'])))







C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Thermal capacity'])

C







C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns=['Air Capacity'])










A = np.zeros([24, 16])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3] = -1                # branch 4: node 3 -> node 4
A[5, 5] = 1                 # branch 5: node 4 -> node 5
A[6, 5], A[6, 6] = -1, 1    # branch 6: node 4 -> node 6
A[7, 6], A[7, 7] = -1, 1    # branch 7: node 5 -> node 6
A[8, 7], A[8, 8] = -1, 1    # branch 8: -> node 7
A[9, 9]= 1                  # branch 9: node 5 -> node 7
A[10, 9], A[10, 10] = -1, 1 # branch 10: -> node 6
A[11, 8], A[11, 10] = -1, 1 # branch 11: -> node 6
A[12, 10], A[12, 12] = -1, 1 #
A[13, 12], A[13, 13] = -1, 1
A[14, 13], A[14, 14] = -1, 1
A[15, 14], A[15, 15] = -1, 1
A[16, 11], A[16, 15] = 1, -1 
A[17, 4] = 1
A[18, 4], A[18, 11] = -1, 1
A[19, 11] = 1
A[20,10], A[20,11] = -1, 1
A[21, 10] = 1
A[22, 10] = 1
A[23, 11] = 1


# np.set_printoptions(suppress=False)
# pd.DataFrame(A)
print (A)















G = np.zeros([24, 24]) 
#wall1_2
a=b=1
G[0,0]=G[5,5]=Gcv_out['wall1_2']
G[1,1]=G[6,6]=G_cd['wall1_2']
G[2,2]=G[7,7]=1/(1/G_cd['layerout1_2'] + 1/G_cd['wall1_2'])
G[3,3]=G[8,8]=G_cd['layerout1_2']
G[4,4]=G[11,11]=Gcv_in['layerout1_2']

#indwall
G[12,12]=Gcv_in['indwall'] 
G[16,16]=Gcv_in['layerout3']   
G[15,15]=G_cd['layerout3']
G[13,13]=G_cd['indwall']
G[14,14]=1/(1/G_cd['layerout3'] + 1/G_cd['indwall'])


#windows

G[17,17]=G[9,9]=Gcv_out['glass']
G[18,18]=G[10,10]=G_cd['glass']

#door   i don t know how it is modelled, conduction or convection?
G[20,20] = Gv              # main door advaction Gcv_out
G[19,19] = Gv              # door between rooms Gcv_in

#ventilation
G[21,21] = Kp

#Controller
G[22,22] = Gv      # controller 1 
G[23,23] = Gv      # controller 2 

#where can i put de long wave radiation convection ?

print('G=',G)


# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
# pd.DataFrame(G)









Cmat=np.zeros([16,16])
Cmat[1,1]=Cmat[6,6]=C['wall1_2']
Cmat[2,2]=Cmat[7,7]=C['layerout1_2']
Cmat[14,14]=C['layerout3']
Cmat[13,13]=C['indwall']

print('Cmat=',Cmat)

# pd.set_option("display.precision", 3)
# pd.DataFrame(C)









b = np.zeros(24)        # branches
b[[0, 5, 9, 17, 19, 21, 22, 23]] = 1   # branches with temperature sources
print(f'b = ', b)









f = np.zeros(16)         # nodes
f[[0, 3, 4, 5, 8, 9, 12, 15]] = 1     # nodes with heat-flow sources
print(f'f = ', f)








y = np.zeros(16)         # nodes
y[[11, 10 ]] = 1              # nodes (temperatures) of interest
print(f'y = ', y)







[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, Cmat, f, y)
print('As = \n', As, '\n')
print('Bs = \n', Bs, '\n')
print('Cs = \n', Cs, '\n')
print('Ds = \n', Ds, '\n')











b = np.zeros(24)        # temperature sources
b[[0, 5, 9, 17, 19]] = 10      # outdoor temperature
b[[21, 22, 23]] = 20            # indoor set-point temperature

f = np.zeros(16)         # flow-rate sources
print(b)









θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print(f'θ = {θ} °C')







bT = np.array([10, 10, 10, 10, 10, 20, 20, 20])     # [To, To, To, Tisp]
fQ = np.array([0, 0, 0, 0, 0, 0, 0, 0])                         # [Φo, Φi, Qa, Φa]
u = np.hstack([bT, fQ])
print(f'u = {u}')







yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
print(f'yss = {yss} °C')






print(f'Max error between DAE and state-space: \
{max(abs(θ[11] - yss)):.2e} °C')








λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As

print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
dtmax = 2 * min(-1. / λ)
print(f'Maximum time step: {dtmax:.2f} s = {dtmax / 60:.2f} min')






# time step
dt = np.floor(dtmax / 60) * 20   # s
print(f'dt = {dt} s = {dt / 60:.0f} min')







# settling time
time_const = np.array([int(x) for x in sorted(-1 / λ)])
print('4 * Time constants: \n', 4 * time_const, 's \n')

t_settle = 4 * max(-1 / λ)
print(f'Settling time: \
{t_settle:.0f} s = \
{t_settle / 60:.1f} min = \
{t_settle / (3600):.2f} h = \
{t_settle / (3600 * 24):.2f} days')









# Step response
# -------------
# Find the next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
n = int(np.floor(duration / dt))    # number of time steps
t = np.arange(0, n * dt, dt)        # time vector for n time steps

print(f'Duration = {duration} s')
print(f'Number of time steps = {n}')
# pd.DataFrame(t, columns=['time'])










u = np.zeros([16, n])                # u = [To To To Tisp Φo Φi Qa Φa]
u[0:5, :] = 10 * np.ones([5, n])    # To = 10 for n time steps
u[5:8, :] = 20 * np.ones([3, n])      # Tisp = 20 for n time steps

# pd.DataFrame(u)









n_s = As.shape[0]                      # number of state variables
θ_exp = np.zeros([n_s, t.shape[0]])    # explicit Euler in time t
θ_imp = np.zeros([n_s, t.shape[0]])    # implicit Euler in time t

I = np.eye(n_s)                        # identity matrix

for k in range(n - 1):
    θ_exp[:, k + 1] = (I + dt * As) @\
        θ_exp[:, k] + dt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (θ_imp[:, k] + dt * Bs @ u[:, k])
    








y_exp = Cs @ θ_exp + Ds @  u
y_imp = Cs @ θ_imp + Ds @  u






fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.T, t / 3600, y_imp.T)
ax.set(xlabel='Time, $t$ / h',
       ylabel='Temperatue, $θ_i$ / °C',
       title='Step input: outdoor temperature $T_o$')
ax.legend(['Explicit', 'Implicit'])
ax.grid()
plt.show()








print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: Room 1: {float(θ[11]):.4f}, Room 2: {float(θ[10]):.4f} °C')
print(f'- state-space model: Room 1: {float(yss[0]):.4f}, Room 2: {float(yss[1]):.4f} °C')
print(f'- steady-state response to step input: Room1: {float(y_exp[0, -2]):.4f}, Room2: {float(y_exp[1, -2]):.4f} °C')








start_date = '01-03 12:00:00'
end_date = '02-05 18:00:00'






start_date = '2000-' + start_date
end_date = '2000-' + end_date
print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')











filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data














weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather.loc[start_date:end_date]
















surface_orientation = {'slope': 90,
                       'azimuth': 0,
                       'latitude': 45}
albedo = 0.2
rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, albedo)
# pd.DataFrame(rad_surf)



















rad_surf['Φtot'] = rad_surf.sum(axis=1)







# resample weather data
data = pd.concat([weather['temp_air'], rad_surf['Φtot']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'To'})
# pd.DataFrame(data)








data['Ti'] = 20 * np.ones(data.shape[0])
data['Qa'] = 0 * np.ones(data.shape[0])
# pd.DataFrame(data)








# input vector
To = data['To']
Ti = data['Ti']
Φo = α_wSW * wall['Surface']['Layer_out'] * data['Φtot']
Φi = τ_gSW * α_wSW * wall['Surface']['Glass'] * data['Φtot']
Qa = data['Qa']
Φa = α_gSW * wall['Surface']['Glass'] * data['Φtot']

u = pd.concat([To, To, To, Ti, Φo, Φi, Qa, Φa], axis=1)
u.columns.values[[4, 5, 7]] = ['Φo', 'Φi', 'Φa']
# pd.DataFrame(u)







θ_exp = 20 * np.ones([As.shape[0], u.shape[0]])








for k in range(u.shape[0] - 1):
    θ_exp[:, k + 1] = (I + dt * As) @ θ_exp[:, k]\
        + dt * Bs @ u.iloc[k, :]
    








    y_exp = Cs @ θ_exp + Ds @ u.to_numpy().T
q_HVAC = Kp * (data['Ti'] - y_exp[0, :])











data['θi_exp1'] = y_exp.T[0,:]
data['θi_exp2'] = y_exp.T[:,1]
data['q_HVAC'] = q_HVAC.T









fig, axs = plt.subplots(2, 1)

data[['To', 'θi_exp1']].plot(ax=axs[0],
                            xticks=[],
                            ylabel='Temperature, $θ$ / °C')
axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'],
              loc='upper right')

data[['Φtot', 'q_HVAC']].plot(ax=axs[1],
                              ylabel='Heat rate, $q$ / W')
axs[1].set(xlabel='Time')
axs[1].legend(['$Φ_{total}$', '$q_{HVAC}$'],
             loc='upper right')
plt.show()









t = dt * np.arange(data.shape[0])   # time vector

fig, axs = plt.subplots(2, 1)
# plot outdoor and indoor temperature
axs[0].plot(t / 3600 / 24, data['To'], label='$θ_{outdoor}$')
axs[0].plot(t / 3600 / 24, y_exp[0, :], label='$θ_{indoor}$')
axs[0].set(ylabel='Temperatures, $θ$ / °C',
           title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600 / 24, data['Φtot'], label='$Φ_{total}$')
axs[1].plot(t / 3600 / 24, q_HVAC, label='$q_{HVAC}$')
axs[1].set(xlabel='Time, $t$ / day',
           ylabel='Heat flows, $q$ / W')
axs[1].legend(loc='upper right')

fig.tight_layout()

















