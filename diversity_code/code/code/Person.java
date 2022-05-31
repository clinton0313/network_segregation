/**
 * An agent in the simulation. Agents have two properties: a minority dummy and a preference ranking.
 */
public class Person {

	private boolean minority;
	private int[] ranking;
	
	/**
	 * Construct an agent.
	 * @param minority whether agent is a minority group member
	 * @param ranking a preference ranking for companies w.r.t. proportional share of minorities employed, and where the ranked item i represents ordered preference of a company with a proportion of minorities closest to the i / (ranking.length - 1) * 100th percentile
	 */
	public Person (boolean minority, int[] ranking) {
		this.minority = minority;
		this.ranking = ranking;
	}
	
	/**
	 * Construct an agent without a preference ranking.
	 * @param minority whether agent is a minority group member
	 */
	public Person (boolean minority) {
		this.minority = minority;
	}
	
	public boolean isMinority () {
		return minority;
	}
	
	public int[] getRanking () {
		return ranking;
	}

}