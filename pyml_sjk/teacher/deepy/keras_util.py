import os.path
from pandas import DataFrame

def train_or_load_weights(model, load_weights=False, **kwargs):
    weights_filepath = kwargs.get('weights_filepath')
    if load_weights and os.path.exists(weights_filepath):
        model.load_weights(weights_filepath)
        return model
    else:
        X_train, y_train = kwargs['X_train'], kwargs['y_train']
        # train
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', int(len(X_train) * 0.02)) # Default 2% of samples
        validation_split = kwargs.get('validation_split', 0.2)
        verbose = kwargs.get('verbose', 1)
        history = model.fit(
            X_train, y_train, epochs=epochs, 
            batch_size=batch_size, shuffle=False, 
            validation_split=validation_split, verbose=verbose)
        # save model
        if weights_filepath is not None:
            model.save_weights(weights_filepath)
        train_result = DataFrame(history.history)
        # save train results
        train_history_filepath = kwargs.get('train_history_filepath')
        if train_history_filepath is not None:
            train_result.to_csv(train_history_filepath, encoding='utf-8')
        return model, train_result
