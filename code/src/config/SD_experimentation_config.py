'''
Config for simulated instances
'''

n_services_range = (900, 1200)
n_services = [i for i in range(n_services_range[0], n_services_range[1]+1, 50)]
scenarios = ["easy", "normal", "hard"]
seeds = range(15)