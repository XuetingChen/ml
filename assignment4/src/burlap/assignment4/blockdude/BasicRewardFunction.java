package burlap.assignment4.blockdude;

import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class BasicRewardFunction implements RewardFunction {

	BlockDudeTF tf;

	public BasicRewardFunction(BlockDudeTF tf) {
		this.tf = tf;
	}

	@Override
	public double reward(State s, GroundedAction a, State sprime) {
		if (tf.isTerminal(sprime)) {
			return 100.;
		}
		return 0;
	}

}
