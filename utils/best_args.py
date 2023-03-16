best_args = {
    'fl_digits': {

        'fedavg': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu': 0.01,
        },

        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'temperature': 0.5,
                'mu':5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        }
    },
    'fl_officecaltech': {

        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 64,
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'temperature': 0.5,
            'mu': 5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        }
    }
}
