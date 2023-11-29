package com.lilium.snake;

import com.lilium.snake.game.Game;
import com.lilium.snake.network.Action;
import com.lilium.snake.network.Environment;
import com.lilium.snake.network.GameState;
import com.lilium.snake.network.util.GameStateUtil;
import com.lilium.snake.network.util.NetworkUtil;
import org.apache.commons.math3.analysis.function.Log;
import org.bytedeco.javacpp.Loader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

public class SnakeDl4j extends JFrame {
    private static final Logger LOG = LoggerFactory.getLogger(SnakeDl4j.class);

    private SnakeDl4j() {
        final Game game = new Game();
        add(game);
        setResizable(false);
        pack();

        setTitle("Deeplearning4j - Snake");
        setLocationRelativeTo(null);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        final Thread thread = new Thread(() -> {
            // Give a name to the network we are about to train
//            final String randomNetworkName = "network-" + System.currentTimeMillis() + ".zip";
            final String randomNetworkName = "network-1624729771113.zip";
            // Create our training environment
            LOG.info("randomNetworkName -> {}", randomNetworkName);
            final Environment mdp = new Environment(game);
            final QLearningDiscreteDense<GameState> dql;
            try {
                UIServer uiServer = UIServer.getInstance();
                StatsStorage statsStorage = new InMemoryStatsStorage();
                uiServer.attach(statsStorage);
                dql = new QLearningDiscreteDense<>(
                        mdp,
//                        DQN.load(randomNetworkName),
                        NetworkUtil.buildDQNFactory(new StatsListener(statsStorage)),
                        NetworkUtil.buildConfig()
                );
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            // Start the training
            dql.train();
            mdp.close();

            // Save network
            try {
                dql.getNeuralNet().save(randomNetworkName);
            } catch (IOException e) {
                LOG.error(e.getMessage(), e);
            }
//            // Reset the game
//            game.initializeGame();
//
//            // Evaluate just trained network
//            evaluateNetwork(game, randomNetworkName);
        });
        thread.start();
    }

    private void evaluateNetwork(Game game, String randomNetworkName) {
        final MultiLayerNetwork multiLayerNetwork = NetworkUtil.loadNetwork(randomNetworkName);
        int highscore = 0;
        for (int i = 0; i < 1000; i++) {
            int score = 0;
            while (game.isOngoing()) {
                try {
                    final GameState state = game.buildStateObservation();
                    final INDArray output = multiLayerNetwork.output(state.getMatrix(), true);
                    double[] data = output.data().asDouble();
                    int maxValueIndex = GameStateUtil.getMaxValueIndex(data);

                    game.changeDirection(Action.getActionByIndex(maxValueIndex));
                    game.move();
                    score = game.getScore();

                    // Needed so that we can see easier what is the game doing
                    NetworkUtil.waitMs(10);
                } catch (final Exception e) {
                    LOG.error(e.getMessage(), e);
                    Thread.currentThread().interrupt();
                    game.endGame();
                }
            }

            LOG.info("Score of iteration '{}' was '{}'", i, score);
            if (score > highscore) {
                highscore = score;
            }

            // Reset the game
            game.initializeGame();
        }
        LOG.info("Finished evaluation of the network, highscore was '{}'", highscore);
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            JFrame ex = new SnakeDl4j();
            ex.setVisible(true);
        });
    }
}
