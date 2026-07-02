best_args = {
    'fl_digits': {

        'fedavg': {
                'local_lr': 0.01,
                'local_batch_size': 32,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 32,
                'mu': 0.01,
        },
        'feddyn': {
                'local_lr': 0.01,
                'local_batch_size': 32,
        },
        'fedproto': {
            'local_lr': 0.01,
            'mu': 1,
            'local_batch_size': 32,
        },
        'fedproc': {
            'local_lr': 0.01,
            'local_batch_size': 32,
        },
        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 32,
                'temperature': 0.5,
                'mu':5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },

        'dlpfl': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedap': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedapc': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedplvm': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'feddap': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'copa': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'feddgga': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE',
            'mu': 0.01,
        },
    },
    'fl_officecaltech': {

        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 32,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 32,
        },
        'feddyn': {
            'local_lr': 0.01,
            'local_batch_size': 32,
        },
        'fedproto': {
            'local_lr': 0.01,
            'mu': 1,
            'local_batch_size': 32,
        },
        'fedproc': {
            'local_lr': 0.01,
            'local_batch_size': 32,
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'temperature': 0.5,
            'mu': 5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },

        'dlpfl': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedap': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedapc': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedplvm': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'copa': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE'
        },
        'fedapw': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'mu': 1,
            'Note': '+ MSE'
        },
        'feddap': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'mu': 1,
            'Note': '+ MSE'
        },
        'feddgga': {
            'local_lr': 0.01,
            'local_batch_size': 32,
            'Note': '+ MSE',
            'mu': 0.01,
        },
    },

    'fl_pacs': {

        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 16,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 16,
        },
        'feddyn': {
            'local_lr': 0.01,
            'local_batch_size': 16,
        },
        'fedproto': {
            'local_lr': 0.01,
            'mu': 1,
            'local_batch_size': 16,
        },
        'fedproc': {
            'local_lr': 0.01,
            'local_batch_size': 16,
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 16,
            'temperature': 0.5,
            'mu': 5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 16,
            'Note': '+ MSE'
        },

        'dlpfl': {
            'local_lr': 0.01,
            'local_batch_size': 16,
            'Note': '+ MSE'
        },

        'fedplvm': {
            'local_lr': 0.01,
            'local_batch_size': 16,
            'Note': '+ MSE'
        },
        'copa': {
            'local_lr': 0.03,
            'local_batch_size': 16,
            'Note': '+ MSE'
        },
        'feddgga': {
            'local_lr': 0.01,
            'local_batch_size': 16,
            'Note': '+ MSE'
        },
        'feddap': {
            'local_lr': 0.01,
            'local_batch_size': 16,
            'mu': 1,
            'Note': '+ MSE'
        },
    },

 
}
