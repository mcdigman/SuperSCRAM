import numpy 


class SURVEYS:
    ''' This class holds and returns information for all surveys
    ''' 
    def __init__(self, surveys):
        self.N_surveys=len(surveys) 
        self.N_tot_obs=0
        for i in range(self.N_surveys):
            self.N_tot_obs=self.N_tot_obs + surveys[i].N_obs
            
        