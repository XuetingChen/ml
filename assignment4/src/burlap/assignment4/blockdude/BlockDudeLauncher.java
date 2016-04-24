package burlap.assignment4.blockdude;

import burlap.assignment4.util.AnalysisAggregator;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;

public class BlockDudeLauncher {
	//These are some boolean variables that affect what will actually get executed
	private static boolean visualizeInitialBlockWorld = false; //Loads a GUI with the agent, walls, and goal
	
	//runValueIteration, runPolicyIteration, and runQLearning indicate which algorithms will run in the experiment
	private static boolean runValueIteration = false;
	private static boolean runPolicyIteration = false;
	private static boolean runQLearning = true;
	
	//showValueIterationPolicyMap, showPolicyIterationPolicyMap, and showQLearningPolicyMap will open a GUI
	//you can use to visualize the policy maps. Consider only having one variable set to true at a time
	//since the pop-up window does not indicate what algorithm was used to generate the map.
	private static boolean showValueIterationPolicyMap = true;
	private static boolean showPolicyIterationPolicyMap = true;
	private static boolean showQLearningPolicyMap = true;
	
	private static Integer MAX_ITERATIONS =100;
	private static Integer NUM_INTERVALS =10;

	public static void main(String[] args) {

		BlockDude bd = new BlockDude();
		Domain domain = bd.generateDomain();
		State initialState = BlockDudeLevelConstructor.getLevel3(domain);

		BlockDudeTF tf = new BlockDudeTF();
		RewardFunction rf = new BasicRewardFunction(tf);

		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf,
				initialState);
		System.out.println("/////Block Dude Analysis/////\n");

		if (visualizeInitialBlockWorld) {
			visualizeInitialBlockWorld(bd, domain, env);
		}
		
		BlockDudeAnalysisRunner runner = new BlockDudeAnalysisRunner(MAX_ITERATIONS,NUM_INTERVALS);
		if(runValueIteration){
			runner.runValueIteration(bd,domain,initialState, rf, tf, showValueIterationPolicyMap);
		}
		if(runPolicyIteration){
			runner.runPolicyIteration(bd,domain,initialState, rf, tf, showPolicyIterationPolicyMap);
		}
		if(runQLearning){
			runner.runQLearning(bd,domain,initialState, rf, tf, env, showQLearningPolicyMap);
		}
		AnalysisAggregator.printAggregateAnalysis();
	}

	private static void visualizeInitialBlockWorld(BlockDude gen, Domain domain,
												  SimulatedEnvironment env) {
		BlockDudeVisualizer vis = new BlockDudeVisualizer();
		Visualizer v = vis.getVisualizer(gen.getMaxx(), gen.getMaxy());
		VisualExplorer exp = new VisualExplorer(domain, env, v);

		exp.initGUI();

	}
	

}
