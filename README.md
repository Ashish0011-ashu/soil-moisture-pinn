# Soil Moisture PINN Project

Yeh ek Physics-Informed Neural Network (PINN) model hai jo soil moisture ko predict karta hai. Basically, ek AI ko sikha rahe hain ki soil moisture real life mein kaise kaam karta hai.

## Yeh Project Kya Hai?

Dekho, humara ek neural network hai jo time ke saath soil moisture levels predict karta hai. Lekin yeh normal neural network nahi hai. Iska matlab yeh actual physics ko samjhta hai!

Model ko pata hai ki:
- Soil moisture random aise hi nahi jump karti
- Aaj ki moisture level kal ke level se related hoti hai (temporal continuity)
- Isse model ko better predictions milte hain

Yeh PINN (Physics-Informed Neural Network) use karta hai, matlab network mein physics rules directly training mein inject kiye hote hain.

## Project Ka Structure


soil-moisture-pinn/
├── models/
│   └── pinn_model.py          # Neural network ka architecture
├── train_pinn.py              # Training script
├── plot_results.py            # Results aur graphs
├── data/                       # Soil moisture data yahan rakhna
└── README.md                  # Yeh file!


## Kaise Chalayein

### 1. Dependencies Install Karo

pip install torch pandas numpy scikit-learn matplotlib


### 2. Apna Data Tayyar Karo
`data/` folder mein apna soil moisture data rakh do. Model ko yeh features chahiye:
- time - timestamp ya time index
- moisture0, moisture1, moisture2, moisture3 - 4 different sensors ke readings
- Model apne aap previous time step ka lag feature bana lega

### 3. Model Ko Train Karo

python train_pinn.py


Yeh karega:
- Data ko 80% training aur 20% validation mein split karega (data leakage nahi hogi)
- Network ko data loss aur physics loss dono se train karega
- Model ko save karega
- Har 50 epochs mein metrics print karega

### 4. Results Dekho

python plot_results.py

Yeh graphs banayega:
- Predictions vs actual values
- Training aur validation loss ke curves
- R² score (model kitna accha hai)

## Technical Details

### Model Ka Architecture


Input (6 features) 
    |
Linear(6 to 64)
    |
LayerNorm(64)  (vanishing gradients se bachata hai)
    |
Tanh (smooth activation)
    |
Linear(64 to 1)
    |
Output (predicted moisture)


**LayerNorm kyun zaruri hai?** Agar yeh nahi hota toh Tanh extreme values pe stuck ho jaata aur gradients vanish ho jaate. LayerNorm sab kuch normalized rakhta hai.

### Loss Function

Yeh interesting part hai. Hum sirf prediction error minimize nahi karte:


Total Loss = Data Loss + (0.05 × Physics Loss)

Jahaan:
- Data Loss = Predictions aur real values ke beech MSE (normal cheez)
- Physics Loss = Previous time step se kitna alag hai prediction

Physics loss se temporal continuity enforce hoti hai. Model seekhta hai ki soil moisture gradually change hoti hai, randomly nahi. Real physics mein moisture teleport nahi ho sakti!

### Chronological Split Kyun Important Hai

Data ko randomly nahi, time ke order mein split karte hain (80% train, 20% test). Isse data leakage nahi hoti jahan model training mein future data dekh le. Real world mein toh sirf forward mein predict karte hain, toh test bhi wahi way mein karte hain.

## Results Ke Metrics

Training ke baad yeh milega:
- R² Score - model kitna accha predict karti hai (1 ke paas toh better)
- RMSE - average prediction error
- Loss curves - training progress

## Hyperparameters Ko Adjust Karo

train_pinn.py mein inn cheezon ko try karo:


lambda_phy = 0.05           # Physics loss ka weight (zyada = strictly physics)
lr = 0.001                  # Learning rate (kam = slow par accurate)
epochs = 500                # Kitne time train kare
patience = 50               # Learning rate reduce karne se pehle wait karo


## Common Issues aur Fixes

**Problem:** Model same value bar bar predict kar raha hai
- Fix: lambda_phy ko badhao physics constraints strict karne ke liye

**Problem:** Loss bahut zyada hai
- Fix: Check karo data normalization aur outliers dekho

**Problem:** Model sikh nahi raha
- Fix: Learning rate kam karo (`lr = 0.0005`) aur longer train karo

## Maine Kya Seekha

1. Physics plus ML together bahut accha kaam karte hain
2. Data leakage rokna bahut zaruri hai (humne hard way se seekha)
3. Layer normalization se kaafi sare problems solve ho jaate hain
4. Temporal continuity real environmental data mein actual cheez hai

## Future Plans

- Attention mechanisms add karne hain
- Different soil types par test karna hai
- Weather features add karne hain (rain, temperature)
- Transfer learning ready banana hai
- API mein deploy karna hai

## Credits

Pytorch se banaya hai kyunki woh best hai. Actual soil science se inspire hua hai.