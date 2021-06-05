import yaml
import pandas as pd
import json
import numpy as np


class yaml_pipeline:
      
    def read_yaml_filename(self,filename,attribute):

        with open(filename) as file:
            decisions = yaml.load(file, Loader=yaml.FullLoader)

        return decisions[attribute]



    # restriction check func

    def restriction_check_nonlist(self,rest,att,data):
        
        min_val=rest['required_values']['min']
        max_val=rest['required_values']['max']
        
        #case1
        if min_val==False and max_val==False:
            return True
        
        #case 2
        elif min_val==False:
            if data[att]<max_val:
                return True
            else:
                return False
            
        #case 3
        elif max_val==False:
            if data[att]>min_val:
                return True
            else:
                return False
        
        #case 4
        else:
            if data[att]>min_val and data[att]<max_val:
                return True
            else:
                return False


    # restriction check func

    def restriction_check_list(self,rest,att,data):
        if data[att] in rest['values']:
            return True
        else:
            return False


    def yaml_parser(self,data,colname,yaml_file):
        
        final_reco=[]

        for m in range(len(data)):

            dict_temp = dict(data[colname])

            if pd.isnull(data[colname][m])==True:
                final_reco.append("no recommendation")

            else:

                list_parse=[]

                dict_temp_1 = dict_temp[m].replace("'", '"')
                j = json.loads(dict_temp_1)
                list_parse = (list(j.keys()))


                for cat in list_parse:

                    check=0

                    #check if the recommendation has some restrictions

                    if cat not in yaml_file:
                        check=1
                        final_reco.append(cat)
                        break

                # if the recommended product has restrictions

                    len_cat = len(yaml_file[cat])
                    c=0

                    for i in range(len_cat):

                        #get the type of restriction whether list or non-list
                        data_type=(yaml_file[cat][i]['type'])

                        if data_type=="non-list":
                            # sending in entire list,column name
                            val=self.restriction_check_nonlist(yaml_file[cat][i],yaml_file[cat][i]['attribute'],data.iloc[m])
                            #print("nonlist:"+str(travel[cat][i]['attribute'])+" "+str(val))
                            if val==1:
                                c=c+1
                                continue
                            else:
                                break

                        elif data_type=="list":
                            # sending in entire list,column name
                            val=self.restriction_check_list(yaml_file[cat][i],yaml_file[cat][i]['attribute'],data.iloc[m])
                            #print("list:"+str(travel[cat][i]['attribute'])+" "+str(val))
                            if val==1:
                                c=c+1
                                continue
                            else:
                                break


                    if c==len_cat:
                        check=1
                        final_reco.append(cat)
                        break

                    else:
                        continue


                if check==0:
                    final_reco.append("no recommendation")
        
        return final_reco