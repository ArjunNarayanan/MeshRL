import optuna


class ParamsExtractorPPO:
    def __init__(self, trial: optuna.Trial, params_config):
        """
        :param params_config: dictionary that is expected to contain optimization
        variables, their type, and their range
        """
        self.trial = trial
        self.params_config = params_config

        self.default_gamma = 1.0
        self.default_n_steps = 1024
        self.default_batch_size = 512
        self.default_gae_lambda = 0.85
        self.default_ent_coef = 1e-4
        self.default_vf_coef = 0.2
        self.default_clip_range = 0.1
        self.default_max_grad_norm = 0.5
        self.default_learning_rate = 1e-4
        self.default_ortho_init = False
        self.default_feature_extractor_layers = 5
        self.default_feature_extractor_size = 512


    def _extract_float(self, key):
        low = self.params_config[key]["low"]
        high = self.params_config[key]["high"]
        log = self.params_config[key].get("log", False)
        val = self.trial.suggest_float(key, low=low, high=high, log=log)
        return val

    def _extract_categorical(self, key):
        choices = self.params_config[key]["choices"]
        return self.trial.suggest_categorical(key, choices=choices)

    def _extract_int(self, key):
        low = self.params_config[key]["low"]
        high = self.params_config[key]["high"]
        step = self.params_config[key].get("step", 1)
        return self.trial.suggest_int(key, low, high, step=step)

    def _extract_value_from_key(self, key, default):
        if key in self.params_config:
            var_type = self.params_config[key]["type"]
            if var_type == "float":
                return self._extract_float(key)
            elif var_type == "categorical":
                return self._extract_categorical(key)
            elif var_type == "int":
                return self._extract_int(key)
            else:
                raise ValueError("Unexpected variable type : ", var_type)
        else:
            return default

    def suggest_parameters(self):

        gamma = self._extract_value_from_key("gamma", self.default_gamma)
        n_steps = self._extract_value_from_key("n_steps", self.default_n_steps)
        batch_size = self._extract_value_from_key("batch_size", self.default_batch_size)
        gae_lambda = self._extract_value_from_key("gae_lambda", self.default_gae_lambda)
        ent_coef = self._extract_value_from_key("ent_coef", self.default_ent_coef)

