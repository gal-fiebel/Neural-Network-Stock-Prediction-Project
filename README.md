# Neural-Network-Stock-Prediction-Project
This is a personal project of mine which I began with the intention of practicing implementing a research paper, using Neural Networks and implementing a ML/ Data Science project.
My project is based on the following research paper: https://www.researchgate.net/publication/319343084_Algorithmic_Trading_Using_Deep_Neural_Networks_on_High_Frequency_Data.
The main purpose of my project is using a simple Neural Network architecture in order to predict the average price of the IBM stock for the next trading day. Below are the steps I took in this project as well as next steps I intend to pursue.

## First Step
My initial objective for this project was implementing the research paper listed above. In order to accomplish this, I needed historical tick by tick trading data (meaning data that is given in milliseconds). However, free data of this sort is not easy to come by. The closest data that I came by to the needed magnitude was the from the following link:
http://api.kibot.com/?action=history&symbol=IBM&interval=1&unadjusted=1&bp=1&user=guest. This is minute by minute trading data of IBM stock from 1998 until this month.
Also, I realized that the algo-trading method in the paper is not suited for my trading needs and capabilities (if my project would yield promising results I would need a strategy that would be feasible for me to apply). For these two reasons I decided to implement the above paper, but to change the minute-by-minute strategy in the paper to a day-by-day strategy. This I accomplished by regarding the inter-minute calculations (linear regression slope for example) as inter-day calculations, meaning the calculations in the paper applied on milliseconds I applied on minutes, thus effectively making minutes into days.
Aside from these changes I performed an almost straight forward implementation of the paper as can be seen in the [two_week_simulation_testing.ipynb]
