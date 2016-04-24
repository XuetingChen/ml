package burlap.assignment4.blockdude;

import burlap.assignment4.util.AnalysisAggregator;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;

import java.util.Arrays;
import java.util.List;

public class BlockDudeAnalysisRunner {

	final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

	private int MAX_ITERATIONS;
	private int NUM_INTERVALS;

	public BlockDudeAnalysisRunner(int MAX_ITERATIONS, int NUM_INTERVALS){
		this.MAX_ITERATIONS = MAX_ITERATIONS;
		this.NUM_INTERVALS = NUM_INTERVALS;
		
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			AnalysisAggregator.addNumberOfIterations(numIterations);

		}

	}
	public void runValueIteration(BlockDude gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		System.out.println("//Value Iteration Analysis//");
		ValueIteration vi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.nanoTime();
			vi = new ValueIteration(
					domain,
					rf,
					tf,
					0.9,
					hashingFactory,
					-1, numIterations); //Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations
										   //for comparison with the other algorithms.
	
			// run planning from our initial state
			p = vi.planFromState(initialState);

			AnalysisAggregator.addMillisecondsToFinishValueIteration((int) (System.nanoTime()-startTime)/1000000);

			// evaluate the policy with one roll out visualize the trajectory
			ea = p.evaluateBehavior(initialState, rf, tf, 100);

			AnalysisAggregator.addValueIterationReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addStepsToFinishValueIteration(ea.numTimeSteps());
		}

		if (showPolicyMap) {
			new EpisodeSequenceVisualizer(BlockDudeVisualizer.getVisualizer(gen.getMaxx(), gen.getMaxx()), domain, Arrays.asList(ea));
		}
		AnalysisAggregator.printValueIterationResults();
		System.out.println("\n\n");
	}

	public void runPolicyIteration(BlockDude gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		System.out.println("//Policy Iteration Analysis//");
		PolicyIteration pi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.nanoTime();
			pi = new PolicyIteration(
					domain,
					rf,
					tf,
					0.9,
					hashingFactory,
					-1, 2, numIterations);
	
			// run planning from our initial state
			p = pi.planFromState(initialState);
			AnalysisAggregator.addMillisecondsToFinishPolicyIteration((int) (System.nanoTime()-startTime)/1000000);

			// evaluate the policy with one roll out visualize the trajectory
			ea = p.evaluateBehavior(initialState, rf, tf, 100);
			AnalysisAggregator.addPolicyIterationReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addStepsToFinishPolicyIteration(ea.numTimeSteps());
		}

		if (showPolicyMap) {
			new EpisodeSequenceVisualizer(BlockDudeVisualizer.getVisualizer(gen.getMaxx(), gen.getMaxx()), domain, Arrays.asList(ea));
		}
		AnalysisAggregator.printPolicyIterationResults();

		System.out.println("\n\n");
	}
	
	public void runQLearning(BlockDude gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf,
			SimulatedEnvironment env, boolean showPolicyMap) {
		System.out.println("//Q Learning Analysis//");

		QLearning agent = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){

			long startTime = System.nanoTime();

			agent = new QLearning(domain, 0.99, hashingFactory, 0.99, 0.99, 10000);
			Policy glie = new EpsilonGreedy(agent, 0.9);
			agent.setLearningPolicy(glie);

			for (int i = 0; i < numIterations; i++) {
				ea = agent.runLearningEpisode(env, 10000);
				env.resetEnvironment();
			}

			agent.initializeForPlanning(rf, tf, 1);
			p = agent.planFromState(initialState);
			AnalysisAggregator.addQLearningReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addMillisecondsToFinishQLearning((int) (System.nanoTime()-startTime)/1000000);
			AnalysisAggregator.addStepsToFinishQLearning(ea.numTimeSteps());
		}
		AnalysisAggregator.printQLearningResults();
		if (showPolicyMap) {
			new EpisodeSequenceVisualizer(BlockDudeVisualizer.getVisualizer(gen.getMaxx(), gen.getMaxx()), domain, Arrays.asList(ea));
		}
		System.out.println("\n\n");

	}
	
	private static List<State> getAllStates(Domain domain,
			 RewardFunction rf, TerminalFunction tf,State initialState){
		ValueIteration vi = new ValueIteration(
				domain,
				rf,
				tf,
				0.99,
				new SimpleHashableStateFactory(),
				.5, 100);
		vi.planFromState(initialState);

		return vi.getAllStates();
	}
	
	public double calcRewardInEpisode(EpisodeAnalysis ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}
	
}
