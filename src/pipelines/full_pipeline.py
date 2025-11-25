# src/pipelines/full_pipeline.py

import mlflow
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from pathlib import Path
import logging

from src.utils.logger import get_logger
from src.models.train import run_train
from src.models.predict import run_predict
from src.models.evaluate import run_evaluate


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ============================================================
    # LOAD ENVIRONMENT VARIABLES (.env)
    # Needed to authenticate MLflow with DagsHub
    # ============================================================
    load_dotenv()

    # ============================================================
    # SETUP
    # ============================================================
    ROOT = Path(get_original_cwd())
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger = get_logger(
        name=cfg.logger.name,
        log_file=ROOT / cfg.logger.file_path,
        level=getattr(logging, cfg.logger.level)
    )

    logger.info("Starting FULL PIPELINE: train -> predict -> evaluate")

    # ============================================================
    # MLflow Setup
    # ============================================================
    mlflow_cfg = cfg.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    logger.info(f"MLflow experiment: {mlflow_cfg.experiment_name}")

    # ============================================================
    # PARENT RUN
    # ============================================================
    with mlflow.start_run(run_name="full_pipeline") as parent_run:

        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent run_id = {parent_run_id}")

        # ----------------------------------------------------------
        # 1) TRAIN STEP
        # ----------------------------------------------------------
        logger.info("STEP 1: Training model...")

        with mlflow.start_run(
            run_name="train_step",
            nested=True
        ) as train_child:

            train_child_id = train_child.info.run_id
            logger.info(f"Train child run_id = {train_child_id}")

            # run_train() DOES NOT open mlflow.run()
            model_run_id, model_path = run_train(cfg, logger)

        # ----------------------------------------------------------
        # 2) PREDICT STEP
        # ----------------------------------------------------------
        logger.info("STEP 2: Predicting test and future forecasts...")

        with mlflow.start_run(
            run_name="predict_step",
            nested=True
        ) as predict_child:

            predict_child_id = predict_child.info.run_id
            logger.info(f"Predict child run_id = {predict_child_id}")

            # run_predict() DOES NOT open mlflow.run()
            test_path, future_path = run_predict(
                cfg,
                logger,
                model_path=model_path,
                parent_run_id=train_child_id,
            )

        # ----------------------------------------------------------
        # 3) EVALUATE STEP
        # ----------------------------------------------------------
        logger.info("STEP 3: Evaluating predictions...")

        with mlflow.start_run(
            run_name="evaluate_step",
            nested=True
        ) as eval_child:

            eval_child_id = eval_child.info.run_id
            logger.info(f"Evaluate child run_id = {eval_child_id}")

            # run_evaluate() DOES NOT open mlflow.run()
            results = run_evaluate(
                cfg,
                logger,
                test_predictions_path=test_path,
                parent_run_id=predict_child_id,
            )

        logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
