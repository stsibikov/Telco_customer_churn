# Telco customer churn

This is a project involving analysis of Telco customer base and building a model to predict the churn of clients.

The project uses my own personal module, designed for preprocessing and EDA, which can be found [here](aku_utils).

### Data

Dataset was taken from [Kaggle](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3). It contains information on customers of a fictional company Telco that provides phone and internet services. The dataset covers customers in the state of California in Q3.

### Goals

* Identify strong and weak sides of services offered by the company, helping the company retain and attract new customers
* Build a machine learning model to predict the churn of customers, helping the company implement targeted retention efforts, including specialized marketing offers.

### Impact

### [Preprocessing](Preprocessing/Preprocessing.ipynb)

Insights from  Preprocessing stage:
* Dataset contains information on:
    * clients themselves
    * their use of services
    * different types of charges associated with said service use
    * satisfaction level
    * churn and reason for churn. This is a very wide range of variables, covering a lot of topics that can enhance our EDA
* The dataset is very clean and did not need any significant processing.
* The dataset contains a benchmark in the form of churn probability inferred by a model built in IBM SPSS Modeler. This is very useful to comprate model effectiveness.

### [EDA](EDA/EDA.ipynb)

During EDA, I looked at customer portrait, use of services and satisfaction score associated with them, as well as churn and churn reasons.

Insights from  EDA stage:

**Customer portrait**

* Our customers are concentrated in larger cities of the state, specifically Los Angeles and San Diego
* They are, however, found throughout the whole state, which means that we, at the very least, have coverage in these places. It would be nice to see how the customer satisfaction and churn depends in these places compared to large cities
* The ages from 28 to 63 are slightly more prevalent in our customer base than state population, which, on one side, is to be expected given the services we offer, but, on the other side, we might be undertargeting the category of young adults, aged 21-28.

**Services**

Services that need improvement, from most impactful to least:
* Unlimited data
* Fiber optic internet
* Streaming TV, movies or music experiences
* Cable internet

Services that need to be pushed and advertised to customers heavier, from most impactful to least:
* Premium tech support
* Online backup
* Device protection plan
* Online security

The biggest reasons for churn are:
* Non-competitive proposals or devices
* Attitude of the support personnel

Specific problems with some of the services:
* Online backup - too much general dissatisfaction with the product
* Unlimited data, Device protection plan - bad or unreliable devices, unstable network etc
* Multiple phone lines - bad, non-competitive offers
* Phone service and movie streaming - poor attitude or expertise of support or service provider

My recommendations regarding services:
* Additional training of support personnel or service providers to ensure proper customer satisfaction
* Upgrading devices to match competitors' levels
* Offer more data and download speeds in offers
* Increasing prices on offers (dissatisfaction level tied to service prices is relatively low)

### Machine Learning

