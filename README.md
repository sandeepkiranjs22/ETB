## ETB Modules/Use-Cases

. Credit Card

. Current Account

. Funds

. Insurance

. Loan

. Savings Account

. Term Deposit


## Basic Terminologies /  Modules used     

. Dask for Parallel Processing

. Every Use case has a up-sell / cross-sell module

. Pre-processing Pipeline

. Model Traininig Pipeline


## Dask for Parallel Processing

. Dask is a parallel computation framework that has seamless integration with your Jupyter notebook. Originally, it was built to overcome the storage limitations of a single machine and extend the computation capability of Pandas, Numpy, and Scit-kit Learn with DASK equivalents, but soon it found its use as a generic distributed system.



### Dask Clusters

Dask networks are composed of three pieces:

. A centralized scheduler, which manages workers and assigns the tasks that need to be completed by them

. Many workers, which do the computation, hold onto results, and communicate results to each other

. One or multiple clients, from which users interact from Jupyter notebooks or scripts and submit work to the scheduler for execution on the workers


The client would be sending the request on what kind of code to compute, the scheduler receives the request and divides the work that needs to be done amongst workers to satisfy the request, and workers finally do all the computational work.

![alt_text](https://miro.medium.com/max/700/0*9JHQAjTVoKbm2f4X.png)

As you can see, Dask breaks up these big data computations into many smaller computations.





### Important Aspects of Dask: 


**Scalability**

Dask scales up Pandas, Scikit-Learn, and Numpy natively with python and Runs resiliently on clusters with multiple cores or can also be scaled down to a single machine.

**Scheduling**

Dask Task Schedulers are optimized for computation much like Airflow, Luigi. It provides rapid feedback, tracks tasks using Task graphs, and aids in diagnostics both in local and distributed mode aking it interactive and responsive.

![alt text](https://miro.medium.com/max/700/1*0OaznYUVfHwDJacwqfSBFg.png)


### Advantages of Dask:

Familiar: Provides parallelized NumPy array and Pandas DataFrame objects

Flexible: Provides a task scheduling interface for more custom workloads and integration with other projects.

Native: Enables distributed computing in pure Python with access to the PyData stack.

Fast: Operates with low overhead, low latency, and minimal serialization necessary for fast numerical algorithms

Scales up: Runs resiliently on clusters with 1000s of cores

Scales down: Trivial to set up and run on a laptop in a single process

Responsive: Designed with interactive computing in mind, it provides rapid feedback and diagnostics to aid humans



### Installation of Dask:

**Conda Installation**

```
conda install dask
```

**Pip Installation**

```
pip install dask[complete]
```


### Starting a Dask Cluster and connecting the Client

For Dask to handle all the computations, the first thing you need to set up is the cluster on which your code is going to run on. To start a local cluster, use the following command:

```
from dask.distributed import LocalCluster, Client

cluster = LocalCluster()
client = Client(cluster)

#To see where the port of the dashboard is, use this command

print(client.scheduler_info()['services'])

# {'dashboard': 8787} --> means you can access it at localhost:8787
```



### Conclusion:

Dask is a fault-tolerant, elastic framework for parallel computation in python that can be deployed locally, on the cloud, or high-performance computers. Not only it scales out capabilities of Pandas and NumPy, but also it can be used as Task schedulers. 


## Pre-Processing Module

. Here , there is a generic template class, which has all the function modules required for pre-processing irrespective of the ETB-Use Case

. Modules present under this are : 

 . Derived Data Calculation

 . Handling Missing Data

 . Outlier Detection

 . Feature Scaling

 . Correlation Analysis

 . Feature Importance 


. Folder Name: preprocessing_pipeline


```
class DataPreprocess:

    def __init__(self, data_file):
        """loads data"""
        df = pd.read_csv(data_file)
        self.df = df
    
    ## function to get the dataframe
    def get_df(self):
        return self.df

    ## function to drop columns
    def drop_column(self,data,col):
        data.drop(col,axis=1,inplace=True)
        return data
        
    ## function to change all column names to lower and eliminate whitespace in column names
    def format_column_name(self, dataset):
        """Changing all column names to lower and eliminate whitespace"""

        with parallel_backend('dask'):
            cols = list(dataset.columns)
            cols = [x.lower().strip() for x in cols]
            cols = [re.sub('[\W_]+', '_', x) for x in cols]
        return cols
    
    # Function to return data type information about dataframe
    def check_dtype(self, dataset):
        """returns a dataframe listing all column names"""
        cols = dataset.columns
        info = []
        temp_dict = {}
        for col in cols:
            temp_dict['column'] = col
            temp_dict['dtype'] = dataset[col].dtype
            temp_dict['sample'] = dataset[col][0]
            info.append(temp_dict)
            temp_dict = {}
        df = pd.DataFrame(info)
        return df

    # Function to return all columns which are of object type
    def get_obj_col_list(self, df):
        obj_cols = []
        for col in df.columns:
            if df[col].dtypes == "object":
                obj_cols.append(col)
        return obj_cols

    # Function to return all columns which are of non-object type
    def get_nonobj_col_list(self, df):
        nonobj_cols = []
        for col in df.columns:
            if df[col].dtypes != "object":
                nonobj_cols.append(col)
        return nonobj_cols

    # Function to convert data to lower case for all categorical columns
    def convert_data_to_lowercase(self, df, col_list):
        """cols_list = categorical cols list to lowercase all entries"""
        for col in col_list:
            df[col] = df[col].str.lower()
        return df

    # Function to return all unique values present in the columns of the dataframe
    def get_unique_col_values(self, dataset):
        """returns """
        cols = dataset.columns
        info = []
        temp_dict = {}
        for col in cols:
            num = len(dataset[col].unique())
            #         if num == 1 or num == len(dataset):
            temp_dict['column'] = col
            temp_dict['num_unique_values'] = num
            info.append(temp_dict)
            temp_dict = {}
        df = pd.DataFrame(info)
        df = df.sort_values('num_unique_values')
        return df

    """CALCULATE DERIVED ATTRIBUTES"""

    # Function to return age from dob column
    def calculate_age(self, born):
        """usage - calculate_age(date(1980, 5, 26))"""
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def get_age_col(self, df, dob_col, split_on, dindex, mindex, yindex):
        """Usage - get_age_col(customer_df, 'dob', '-')"""
        with parallel_backend('dask'):
            df['age'] = ""
            for i in range(len(df)):
                date_ = df[dob_col][i]
                date_ = date_.split(split_on)
                year = int(date_[yindex]) #0
                month = int(date_[mindex]) #1
                day = int(date_[dindex]) #2
                df['age'][i] = int(self.calculate_age(date(year, month, day)))
            df['age'] = pd.to_numeric(df['age'])
            df.drop(dob_col, axis=1, inplace=True)
            return df

    # Function to calculate time span of customer
    def calculate_customer_since(self, oldest_trans):
        today = date.today()
        years = today.year - oldest_trans.year - ((today.month, today.day) < (oldest_trans.month, oldest_trans.day))
        return years * 12 + today.month - oldest_trans.month
    

    def get_customer_since(self, df, oldest_trans_col, split_on, dindex, mindex, yindex):
        """"""
        """Usage - get_customer_since(customer_df, 'date_first_trans', '/')"""
        with parallel_backend('dask'):
            df['customer_since_months'] = ""

            for i in range(len(df)):
                date_ = df[oldest_trans_col][i]
                date_ = date_.split(split_on)
                year = int(date_[yindex]) #2
                month = int(date_[mindex]) #0
                day = int(date_[dindex]) #1
                df['customer_since_months'][i] = int(self.calculate_customer_since(date(year, month, day)))
                df['customer_since_months'] = pd.to_numeric(df['customer_since_months'])
            df.drop(oldest_trans_col, axis=1, inplace=True)
            return df

    def get_months(self, df, date_col, split_on, dindex, mindex, yindex):
        """"""
        """Usage - get_customer_since(customer_df, 'date_first_trans', '/')"""
        colname = 'months_ago_' + date_col
        with parallel_backend('dask'):
            df[colname] = ""

            for i in range(len(df)):
                date_ = df[date_col][i]
                date_ = date_.split(split_on)
                year = int(date_[yindex])
                month = int(date_[mindex])
                day = int(date_[dindex])
                df[colname][i] = int(self.calculate_customer_since(date(year, month, day)))
                df[colname] = pd.to_numeric(df[colname])
            df.drop(date_col, axis=1, inplace=True)
            return df


    #zip code -> lat , long -> finding dist between address and bank

    # Function to calculate distance from customer address to bank using pre-defined apis which computes distance
    def zipcode_distance_retrieval(self,col,country):
        
        list_dist=[]
        
            
        address = col

        address = address.split(",")

        """USAGE - get zipcode of place("Whitefield")"""

        place = address[0].title()
        city  = address[1].title()
        place_dict = {  "adminName2": city,
                         "placeName": place}

        q = 'https://parseapi.back4app.com/classes/Worldzipcode_'+country+'?limit=1&excludeKeys=accuracy,adminCode1,adminCode2&where=' + json.dumps(
          place_dict)
        query = json.dumps(q)

        response = requests.get(json.loads(query),

                                headers={
                                    'X-Parse-Application-Id': 'GSNkm3CSoxzgwo47wCdufbaDFA3SvgqRh9hJ82mt',
                                    'X-Parse-REST-API-Key':   'brIpMJbLmH565ZiY1ODhpLX08cO4N9jdqiedqAHY'
                                })

        data_res = response.json()


        if response.status_code!=404:

            result  = data_res['results'][0]['geoPosition']
            zipcode = data_res['results'][0]['postalCode']

            zip_1 = (result['latitude'],result['longitude'])

            # fix position of zip code 2

            zip_2 = (12.9698,77.7500)

            dist =round(mpu.haversine_distance(zip_1,zip_2),2)

            #print(dist)

            
            return dist

        else:
            return 0


    def address_distance(self,data,dict_add,col):
        
        list_add = dict_add.keys()
        
        dist=[]
        
        for i in range(len(data)):
            
            if int(dict_add[data[col][i]]) < 15:
                dist.append(1)
            else:
                dist.append(0)
            
        data.drop(col,axis=1,inplace=True)
        
        
        data["address_near_by_bank"] =dist
        
        return data    


    # computation of mean , variance , standard deviation, skeweness , peak

    def stats_return(self,data,col1,col2):
    
        list_val=[]

        for index in range(len(data)):
        # calculate mean

            n= len(col1)
            sum=0
            for cols in col1:
                sum = sum + data[cols][index]


            mean=(sum/n)

            #print('mean: ', mean)

            #  calculate variance

            sum2=0

            for cols in col1:
                sum2=sum2+ (data[cols][index]-mean)**2

            var=sum2/n

            #print("Variance: ", var)


            # calculate median

            if n % 2 == 0:
                t1 = col1[n//2]
                t2 = col1[n//2 -1 ]
                median1 = data[t1][index] 
                median2 = data[t2][index]
                median = (median1 + median2)/2

            else: 
                median = data[t1][index]

            #print("Median : " ,median) 


            # calculate sd

            sd = math.sqrt(var)

            #print("Standard dev:",sd)

            # calculate skewness
            skew = 3 * ( mean - median)
            skew = skew / sd

            #print("skewness:",skew)

            # calculate peak

            peak = 0

            for cols in col1:
                peak=max(peak,data[cols][index])

            #print("peak value is : ",peak)




            # column2


            # calculate mean

            n= len(col2)
            sum=0
            for cols in col2:
                sum = sum + data[cols][index]


            mean2=(sum/n)

            #print('mean: ', mean)

            #  calculate variance

            sum2=0

            for cols in col2:
                sum2=sum2+ (data[cols][index]-mean2)**2

            var2=sum2/n

            #print("Variance: ", var)


            # calculate median

            if n % 2 == 0:
                t1 = col2[n//2]
                t2 = col2[n//2 -1 ]
                median1 = data[t1][index] 
                median2 = data[t2][index]
                median = (median1 + median2)/2

            else: 
                median = data[t1][index]

            #print("Median : " ,median) 


            # calculate sd

            sd2 = math.sqrt(var2)

            #print("Standard dev:",sd)

            # calculate skewness
            skew2 = 3 * ( mean2 - median)
            skew2 = skew2 / sd2

            #print("skewness:",skew)

            # calculate peak

            peak2 = 0

            for cols in col2:
                peak2=max(peak2,data[cols][index])

            #print("peak value is : ",peak)


            # if skewness is poistive and mean trans becomes less atleast for 4 months 

            count1=0
            count2=0



            for cols in col1:
                if mean>data[cols][index]:
                    count1=count1+1

            for cols in col2:
                if mean2>data[cols][index]:
                    count2=count2+1

            if count1>=4 and count2>=4 and skew>0 and skew2>0:
                list_val.append(0.8)

            elif count1>4 and skew>0 or count2>4 and skew2>0:
                list_val.append(0.5)

            else:
                list_val.append(0.3)


        data['transactions_churn_analysis']=list_val
        
        return data


    # Function to find  single limit and total limit depending on CSL ( Example: 25/100 )
    def policy_csl_func(self,data,col,split_on):
    
        data['bodily_injury_single_limit']=""
        data['bodily_injury_total_limit']=""
        
        list1=[]
        list2=[]
        
        for i in range(len(data)):
            csl = data[col][i].split(split_on)
            list1.append(int(csl[0]))
            list2.append(int(csl[1]))
        
        data['bodily_injury_single_limit']=list1
        data['bodily_injury_total_limit']=list2
        
        data.drop(col,axis=1,inplace=True)
        
        return data

    # Function to calculate the total net captial
    def net_amt(self,data,col1,col2):
    
        data['net_captial']=""
        net_list=[]
        
        for i in range(len(data)):
            net_list.append(data[col1][i]+data[col2][i])
            
        data.drop([col1,col2],axis=1,inplace=True)
        
        data['net_captial']=net_list
        
        return data


    # Function to calculate Severity and assign numbers to the categorical columns
    def severity_ordinal(self,data,col1):
    
        col_list=[]
        
        for i in range(len(data)):
            
            if data[col1][i]=='Total Loss':
                col_list.append(0.8)
                
            elif data[col1][i]=='Major Damage':
                col_list.append(0.6)
                
            elif data[col1][i]=="Minor Damage":
                col_list.append(0.4)
            
            else:
                col_list.append(0.2)
    
        data[col1]=col_list
        
        return data

    
    # Function to convert time to morning/evening/night
    def hour_segment(self,data2,col):
        
        
        segment=[]
        
        for i in range(len(data2)):
            hour=data2[col][i]
            # morning ( 04 -12 )
            if hour>=4 and hour<12:
                segment.append("morning")

            elif hour>=12 and hour<19:
                segment.append("evening")

            else:
                segment.append("night")

        data2[col]=segment
        
        return data2

    # Function used for bucket segmenting
    def witness_bucket_segment(self,data,col):
    
        val=[]
        for i in range(len(data)):
            if data[col][i]>=2:
                val.append(2)
            else:
                val.append(data[col][i])

        data[col]=val
        
        return data


    # Function to convert cars of customer to ordinal values
    def auto_make_to_ordinal(self,data,col1,col2):
        
        prem_cars_list=['Mercedes','BMW','Audi']
        
        other_cars_list=['Saab',  'Dodge', 'Chevrolet', 'Accura', 'Nissan', 'Toyota', 'Ford', 'Suburu',
                    'Jeep','Honda','Volkswagen']
        
        auto_make=[]
        
        for i in range(len(data)):
            
            if data[col1][i] in prem_cars_list:
                auto_make.append(10000 * data[col2][i])
            else:
                auto_make.append(4000 * data[col2][i])
        
        data[col1]=auto_make
        
        return data
    

    # transaction analysis function
    # computation of mean , variance , standard deviation, skeweness , peak

    def stats_return(self,data,col1,col2):
        
        list_val=[]

        for index in range(len(data)):
        # calculate mean

            n= len(col1)
            sum=0
            for cols in col1:
                sum = sum + data[cols][index]


            mean=(sum/n)

            #print('mean: ', mean)

            #  calculate variance

            sum2=0

            for cols in col1:
                sum2=sum2+ (data[cols][index]-mean)**2

            var=sum2/n

            #print("Variance: ", var)


            # calculate median

            if n % 2 == 0:
                t1 = col1[n//2]
                t2 = col1[n//2 -1 ]
                median1 = data[t1][index] 
                median2 = data[t2][index]
                median = (median1 + median2)/2

            else: 
                median = data[t1][index]

            #print("Median : " ,median) 


            # calculate sd

            sd = math.sqrt(var)

            #print("Standard dev:",sd)

            # calculate skewness
            skew = 3 * ( mean - median)
            skew = skew / sd

            #print("skewness:",skew)

            # calculate peak

            peak = 0

            for cols in col1:
                peak=max(peak,data[cols][index])

            #print("peak value is : ",peak)




            # column2


            # calculate mean

            n= len(col2)
            sum=0
            for cols in col2:
                sum = sum + data[cols][index]


            mean2=(sum/n)

            #print('mean: ', mean)

            #  calculate variance

            sum2=0

            for cols in col2:
                sum2=sum2+ (data[cols][index]-mean2)**2

            var2=sum2/n

            #print("Variance: ", var)


            # calculate median

            if n % 2 == 0:
                t1 = col2[n//2]
                t2 = col2[n//2 -1 ]
                median1 = data[t1][index] 
                median2 = data[t2][index]
                median = (median1 + median2)/2

            else: 
                median = data[t1][index]

            #print("Median : " ,median) 


            # calculate sd

            sd2 = math.sqrt(var2)

            #print("Standard dev:",sd)

            # calculate skewness
            skew2 = 3 * ( mean2 - median)
            skew2 = skew2 / sd2

            #print("skewness:",skew)

            # calculate peak

            peak2 = 0

            for cols in col2:
                peak2=max(peak2,data[cols][index])

            #print("peak value is : ",peak)


            # if skewness is poistive and mean trans becomes less atleast for 4 months 

            count1=0
            count2=0



            for cols in col1:
                if mean>data[cols][index]:
                    count1=count1+1

            for cols in col2:
                if mean2>data[cols][index]:
                    count2=count2+1

            if count1>=4 and count2>=4 and skew>0 and skew2>0:
                list_val.append(0.8)

            elif count1>4 and skew>0 or count2>4 and skew2>0:
                list_val.append(0.5)

            else:
                list_val.append(0.3)


        data['transactions_churn_analysis']=list_val
        
        return data

    # function to get number of days elapsed from current date
    def get_days_elapsed_from_today(self,data,col,split_on, d,m,y):

        list_days=[]
        
        for i in range(len(data)):

            date = data[col][i]
            
            date = date.split(split_on)

            year = int(date[y])

            month = int(date[m])

            day = int(date[d])

            today = datetime.date.today()

            day_taken=datetime.date(year,month,day)

            diff = today-day_taken

            list_days.append(diff.days)
            
        data[col+"_in_days"]=list_days
        
        data.drop(col,axis=1,inplace=True)
        
        return data
        

    # Function to calculate Span in years
    def get_years_old(self,data,col):
        
        data['years_auto_old']=""
        
        years_old=[]
        
        today = datetime.datetime.now()
        
        for i in range(len(data)):
            years_old.append(today.year-data[col][i])
        
        data['years_auto_old']=years_old
        
        data.drop(col,axis=1,inplace=True)
        
        return data


    def get_vehicle_age(self, df, vehicle_bought_in_col):
        df['vehicle_age_yrs'] = 0
        now = datetime.datetime.now()
        with parallel_backend('dask'):
            for i in range(len(df)):
                vage = now.year - int(df[vehicle_bought_in_col][i])
                df['vehicle_age_yrs'][i] = vage
        df.drop(vehicle_bought_in_col, axis=1, inplace=True)
        return df


    # Function to Segment customers based on customer relationship time with bank
    def group_customer(self, df, customer_since_months_col, customer_rel_dur_segment):
        """"""
        """Usage - group_customer(customer_df, 'customer_since_months', 'customer_rel_dur_segment')"""
        """takes in the customer_since_months col to group customers into old and new"""
        df[customer_rel_dur_segment] = ""
        for i in range(len(df)):
            num_months = df[customer_since_months_col][i]
            if num_months > 24:
                df[customer_rel_dur_segment].iloc[i] = "old_customer"
            else:
                df[customer_rel_dur_segment].iloc[i] = "new_customer"
        return df

    def get_product_bought_in_month(self, df, date_col, month_index, split_on):
        df['product_bought_in_month'] = ""

        for i in range(len(df)):
            date_ = df[date_col][i]
            date_ = date_.split(split_on)
            month = int(date_[month_index])
            df['product_bought_in_month'][i] = month
            df['product_bought_in_month'] = df['product_bought_in_month'].astype(str)
        df.drop(date_col, axis=1, inplace=True)
        return df


    # Function to find the population of the city of customer
    def get_population(self, city):
        """USAGE - get_population("Mumbai")"""
        city = city.title()
        city_dict = {"name": city}

        q = 'https://parseapi.back4app.com/classes/Continentscountriescities_City?limit=1&where=' + json.dumps(
            city_dict)
        query = json.dumps(q)

        response = requests.get(json.loads(query),

                                headers={
                                    'X-Parse-Application-Id': 'A9ccOjqi9naJzrBSAWWtPWobfZC5MGJQ04JZfKEc',
                                    'X-Parse-REST-API-Key': 'CIBl0NfLJ9uHTWgPtkDSygklg3ht3kTFfB9bJdPQ'
                                })

        data = response.json()
        population = data.get('results')[0].get('population')
        return population

    def get_population_column(self, df, city_col):
        df['population'] = 0
        cities = list(df[city_col].unique())
        city_pop_dict = {}

        with parallel_backend('dask'):
            for city in cities:
                try:
                    pop = self.get_population(city)
                except:
                    pop = 0
                city_pop_dict[city] = pop

            for i in range(len(df)):
                city = df[city_col][i]
                df['population'][i] = int(city_pop_dict[city])
                df['population'] = pd.to_numeric(df['population'])
            df.drop(city_col, axis=1, inplace=True)
        return df

    def get_population_bins(self, df, population_col):
        bins_for_population = df[population_col].quantile([0, 0.25, 0.5, 0.75, 1])
        bins_dict = {}
        bins_dict[0] = bins_for_population[0]
        bins_dict[0.25] = bins_for_population[0.25]
        bins_dict[0.5] = bins_for_population[0.5]
        bins_dict[0.75] = bins_for_population[0.75]
        bins_dict[1] = bins_for_population[1]
        return bins_dict

    def get_labels(self, bin_array):
        labels = []
        for i in range(len(bin_array)-1):
            """Bin labels must be one fewer than the number of bin edges"""
            text = 't'
            label = text + str(i)
            labels.append(label)
        return labels

    def group_city_from_population(self, df, pop_col, bins, labels):
        df['city_grp'] = pd.cut(df[pop_col], bins=bins, labels=labels)
        df['city_grp'] = df['city_grp'].cat.add_categories('unknown')
        df['city_grp'].fillna('unknown', inplace=True)
        df['city_grp'] = df['city_grp'].astype(str)
        df.drop(pop_col, axis=1, inplace=True)
        return df

    """HANDLE MISSING DATA"""

    ### Function to give details on the number of missing values in every column	
    def get_missing_df(self, df):
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        return missing_value_df

    ### Function for handling missing values in object datatype	
    def object_datatype_missing_val(self, dataset, list_nonobject, col):
        """handling missing values in object type columns of the dataset"""
        """parameter - col is the column to be imputed"""

        list_cols_null = dataset.columns[dataset.isna().any()].tolist()
        X = dataset[list_nonobject]
        for i in list_cols_null:
            if i in X.columns:
                X.drop(i, axis=1, inplace=True)
        y = dataset[col]
        z = pd.DataFrame(y)

        # with null values
        test_indx = z[z[col].isnull() == True]
        # without null
        train_indx = z[z[col].isnull() == False]

        test_list = list(test_indx.index)
        train_list = list(train_indx.index)
        y_train = train_indx
        y_test = test_indx
        x_train = X.iloc[train_list, :]
        x_test = X.iloc[test_list, :]

        """using ExtraTreesClassifier to get most important feature."""
        extra_tree_forest = ExtraTreesClassifier(n_estimators=5, criterion='entropy', max_features=2)
        print(x_train.shape)
        print(y_train.shape)

        """Training the model"""
        extra_tree_forest.fit(x_train, y_train)
        feature_importance = extra_tree_forest.feature_importances_
        feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                                extra_tree_forest.estimators_], axis=0)

        d = dict(zip(list_nonobject, feature_importance_normalized))
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        """taking top 3 features"""
        features_list = list(d.keys())[:3]
        print(features_list)
        print("-----------")

        x_train = x_train[features_list]
        x_test = x_test[features_list]
        knn = KNeighborsClassifier(n_neighbors=7).fit(x_train, y_train)
        # putting back into dataframe
        knn_predictions = knn.predict(x_test)
        i = 0
        for val in test_list:
            dataset[col].iloc[val] = knn_predictions[i]
            i = i + 1
        return dataset[col]

    def missing_data_handle(self, dataset, missing_value_df, list_nonobject):
        with parallel_backend('dask'):
            # Correaltion matrix
            corr_matrix = dataset.corr()
            imp_neighbour = KNNImputer(n_neighbors=2, weights="uniform")
            imputer = KNNImputer()

            with parallel_backend('dask'):

                """non object data type ( int 64, float 64 )"""
                list_nonobject_null = []
                cols_temp2 = []

                for i in range(len(missing_value_df)):
                    if missing_value_df['percent_missing'][i] > 0:
                        cols_temp2.append(missing_value_df['column_name'][i])

                for i in cols_temp2:
                    if dataset[i].dtypes != object:
                        list_nonobject_null.append(i)

                for col in list_nonobject_null:
                    dict_corr_matrix = dict(corr_matrix[col].sort_values(ascending=False))
                    out = dict(list(dict_corr_matrix.items())[0:5])
                    correlated_cols = out.keys()
                    dataset[col] = imp_neighbour.fit_transform(dataset[correlated_cols])

                """object data type"""
                list_objects_null = []
                cols_temp1 = []

                for i in range(len(missing_value_df)):
                    if missing_value_df['percent_missing'][i] > 0:
                        cols_temp1.append(missing_value_df['column_name'][i])

                for i in cols_temp1:
                    if dataset[i].dtypes == object:
                        list_objects_null.append(i)

                for col in list_objects_null:
                    dataset[col] = self.object_datatype_missing_val(dataset, list_nonobject, col)
        return dataset

    """HANDLING CATEGORICAL DATA"""

    ### Functions to handle categorical data columns and missing values in categorical columns 

    def get_cat_cols_unique_val_dict(self, df, obj_cols):
        """returns a dict with columns grouped by num_of_unique_values in them"""
        col_uniques_df = self.get_unique_col_values(df[obj_cols])
        col_uniques_df = dict(col_uniques_df.groupby('num_unique_values')['column'].apply(lambda x: "%s" % ",".join(x)))
        return col_uniques_df

    def group_less_occurring_cat_vars(self, dataset, obj_cols):
        """grouping less occurring categorical variables into others category"""
        with parallel_backend('dask'):
            cols_handle = []
            for col in obj_cols:
                cols_freq = dict(dataset[col].value_counts(normalize=True) * 100)
                # highest = list(cols_freq.values())[0]
                items = list(cols_freq.items())
                vals = []

                for i in range(len(items)):
                    if items[i][1] <= 0.1:
                        vals.append(items[i][0])

                for i in range(len(dataset)):
                    if dataset[col][i] in vals:
                        dataset[col][i] = "others"
        return dataset

    def convert_cat_cols_to_binary(self, dataset, binary_cols_list):
        for col in binary_cols_list:
            temp = {}
            values = dataset[col].unique()
            count = 0
            for value in values:
                temp[value] = count
                count += 1
            dataset[col].replace(temp, inplace=True)
            temp = {}
        return dataset

    def convert_cat_cols_to_onehot(self, dataset, onehot_col_list):
        # dask's encoder used here
        de = DummyEncoder()
        dataset_addons = de.fit_transform(dataset[onehot_col_list])
        dataset.drop(dataset[onehot_col_list], axis=1, inplace=True)
        dataset = pd.concat([dataset, dataset_addons], axis=1)
        return dataset

    """OUTLIER DETECTION"""

    ### Functions which detailed information about outliers and outerlier detections
	
    def outlier_detection(self, df):
        v = df.values
        mask = np.abs((v - v.mean(0)) / v.std(0)) > 2
        df = pd.DataFrame(np.where(mask, np.nan, v), df.index, df.columns)
        return df

    def outlier_details(self, df):
        """function to get number of outliers column-wise"""
        list_col_name = []
        list_outlier = []

        for i in df.columns:
            list_col_name.append(i)
            list_outlier.append(df[i].isna().sum())

        temp_df = pd.DataFrame()
        temp_df['column'] = list_col_name
        temp_df['outlier'] = list_outlier

        return temp_df

    """FEATURE SCALING"""

    """Feature Scaling - dask functions used for scaling except log and custom scaler"""
 
    ### Here , there are functions for standard scaling , log scaling and min max scaling etc
	
    def standard_scaling(self, df, cols_list):
        scaler = StandardScaler()
        scaler.fit(df[cols_list])
        for col in cols_list:
            scaler.transform(df[col])
        return df

    def log_scaling(self, df, cols_list):
        for col in cols_list:
            df[col] = np.log10(df[col])
        return df

    def minmax_scaling(self, df, cols_list):
        scaler = MinMaxScaler()
        scaler.fit(df[cols_list])
        for col in cols_list:
            scaler.transform(df[col])
        return df

    def custom_scaler(self, df, cols_list, lower, upper):
        """scaled_value = ((value - min) / (max - min)) * (upper - lower)"""
        for col in cols_list:
            min_value = int(df[col].min())
            max_value = int(df[col].max())
            range_ = upper - lower
            for row in range(len(df)):
                temp = df[col][row]
                df[col][row] = ((temp - min_value) / (max_value - min_value)) * range_
        return df

    def plot_col_distribution(self, df, col, bins):
        """function to plot histogram (value distribution) for numerical columns in the dataset"""
        df.hist(column=col, bins=bins)

    """CORRELATION ANALYSIS"""

    def plot_corr(self, df):
        with parallel_backend('dask'):
            corr = df.corr()
            sns.heatmap(corr)

    """get top correlated attributes"""

    def get_redundant_pairs(self, df):
        with parallel_backend('dask'):
            pairs_to_drop = set()
            cols = df.columns
            for i in range(0, df.shape[1]):
                for j in range(0, i + 1):
                    pairs_to_drop.add((cols[i], cols[j]))
            return pairs_to_drop

    def get_top_abs_correlations(self, df, n=5):
        with parallel_backend('dask'):
            au_corr = df.corr().unstack()
            labels_to_drop = self.get_redundant_pairs(df)
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            return au_corr[0:n]

    def drop_highly_corr_var(self, df, corr_threshold):
        """returns a list of columns that can be considered for drop """

        """correlation matrix"""
        corr_matrix = df.corr().abs()

        """Select upper triangle of correlation matrix"""
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        """Find index of feature columns with correlation greater than provided threshold"""
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        #     new_df = df.drop(df[to_drop], axis=1)
        return to_drop

    """Feature Importance"""


    def xgb_feature_importance(self, X_train, y_train, X_test, y_test, param, rounds):
        """X_train etc. coming from a train_test_split function."""

        """xgb classifier to perform feature selection."""

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        param = param
        num_round = rounds

        bst = xgb.train(param, dtrain, num_round)

        ax = xgb.plot_importance(bst, height=0.8, max_num_features=9)
        ax.grid(False, axis="y")
        ax.set_title('Estimated feature importance')
        plt.savefig('feature_importance.png')
        print("feature_imp plot saved")
        return bst
```

## Model Training Module

. Here , there is a generic template class , which has all the function modules required for model training irrespective of the ETB-Use Case

. Folder Name: model_training


```
class ModelTraining:

    def __init__(self, data_file):
        df = pd.read_csv(data_file)
        self.df = df

    # Function 	to get back the dataframe
    def get_df(self):
        return self.df

    # Function 	to convert pandas dataframe to dask dataframe
    def convert_df_to_ddf(self, df):
        x = dd.from_pandas(df, chunksize=50000)
        return x
	

    # Function to get a dictionary for a column value and see if it crosses the given threshold
    def dict_mapper(self,listc,name,threshold):
              
        list1=[]
        
        for i in range(len(listc)):
            
            d=dict()
            
            val = listc[i]
        
            if val>threshold:
                d[name]=(listc[i])
                
            list1.append(d)
        
        return list1


    # Function 	to get convert a binary target column (yes/no) to (1/0)
    def target_col(self,df,y):
        """ target column"""
        for i in range(len(df)):
            if df[y][i]=="yes":
                df[y][i]=1
            else:
                df[y][i]=0
            
        return df	

	
    
    
    # Function 	to encode columns in the dataframe , here send all columns other than target columns
    def label_encoding(self, y):
        """df should not contain the target column"""
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y


    # Function to convert target pandas column to target dask dataframe column
    def convert_target_col_to_dask_df(self, y):
        darr = dd.from_array(y)
        y = darr.to_frame()
        return y

    # Function used for training dask-xgboost algorithm on training dataframe
    def dxgb_train(self, client, params, x, y):
        bst = dxgb.train(client, params, x, y)
        return bst


    # Function 	to get back the list of all columns which are of non-object type
    def get_nonobj_col_list(self, df):
        nonobj_cols = []
        for col in df.columns:
            if df[col].dtypes != "object":
                nonobj_cols.append(col)
        return nonobj_cols

    # Function 	to split dataset into training and testing
    def split_dataset(self, x, y, test_size=0.20):
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=123, test_size=test_size)
        return X_train, X_test, y_train, y_test


    # Function for softmax
    def decode_softmax_to_label(self, prediction_array, reco_mapper, num):
        indexes = sorted(range(len(prediction_array)), key=lambda i: prediction_array[i], reverse=True)[:num]
        if num == 1:
            return indexes[0]

        reco_dict = {}
        for i in indexes:
            reco_dict[reco_mapper[i]] = prediction_array[i]
        return reco_dict
```
