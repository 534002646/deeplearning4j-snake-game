package com.lilium.snake.network.util;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.nd4j.linalg.learning.config.RmsProp;

import java.io.File;
import java.io.IOException;

/**
 * Util class containing methods to build the neural network and its configuration.
 *
 * @author mirza
 */
public final class NetworkUtil {
    /**
     * Number of neural network inputs.
     */
    public static final int NUMBER_OF_INPUTS = 4;
    /**
     * Lowest value of the observation (e.g. player will die -1, nothing will happen 0, will move closer to the food 1)
     */
    public static final double LOW_VALUE = -1;
    /**
     * Highest value of the observation (e.g. player will die -1, nothing will happen 0, will move closer to the food 1)
     */
    public static final double HIGH_VALUE = 1;

    private NetworkUtil() {}

    public static QLearningConfiguration buildConfig() {
        return QLearningConfiguration.builder()
                .seed(123L)
                .maxStep(15000)             // 最大训练步数
                .maxEpochStep(200)          // 最大获取样本步数
                .expRepMaxSize(150000)      // 样本总数量
                .batchSize(128)             // 更新样本量间隔
                .targetDqnUpdateFreq(500)   // 保存神经网络间隔（步数%）
                .updateStart(10)            // 更新样本量起始间隔
                .rewardFactor(0.1)          // 奖励系数
                .gamma(0.99)
                .errorClamp(1.0)
                .minEpsilon(0.1f)
                .epsilonNbStep(1000)
                .doubleDQN(true)
                .build();
    }

    public static DQNFactoryStdDense buildDQNFactory() {
        final DQNDenseNetworkConfiguration build = DQNDenseNetworkConfiguration.builder()
                .l2(0.001)
                .updater(new RmsProp(0.000025))
                .numHiddenNodes(300)
                .numLayers(2)
                .build();

        return new DQNFactoryStdDense(build);
    }

    public static MultiLayerNetwork loadNetwork(final String networkName) {
        try {
            return MultiLayerNetwork.load(new File(networkName), true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * Used to slow down game step so that the user can see what is happening.
     *
     * @param ms Number of milliseconds for how long the thread should sleep.
     */
    public static void waitMs(final long ms) {
        if (ms == 0) {
            return;
        }

        try {
            Thread.sleep(ms);
        } catch (final InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
