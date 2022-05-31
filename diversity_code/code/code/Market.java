import java.util.List;
import java.util.ArrayList;

/**
 * A market is an artificial job market of companies and agents.
 */
public class Market {

	private List<Company> companies;
	private List<Person> employees;
	
	/**
	 * Construct a job market.
	 * @param employees agents to be matched to companies
	 * @param nrCompanies number of companies
	 * @param capacity number of employees each company can hold
	 */
	public Market (List<Person> employees, int nrCompanies, int capacity) {
		this.employees = employees;
		Company.setCapacity(capacity);
		companies = new ArrayList<Company>(nrCompanies);
		for (int i = 0; i < nrCompanies; i++)
			companies.add(new Company());
	}
	
	/**
	 * Add an agent to the market.
	 */
	public void addEmployee (Person employee) {
		employees.add(employee);
	}
	
	/**
	 * Assign an agent to a specified company.
	 * @return whether employee could be added
	 */
	public boolean engage (Person employee, Company company) {
		return company.addEmployee(employee);
	}
	
	/**
	 * Assign an agent to the company that has a specified position in the list
	 * @return whether employee could be added
	 */
	public boolean engage (Person employee, int companyNr) {
		return companies.get(companyNr).addEmployee(employee);
	}
	
	/**
	 * Find the company with remaining capacity that is the closest match to an agent's preference ranking.
	 * @param employee the agent for which to find the optimal match
	 * @return the company that is the best match
	 */
	public Company match (Person employee) {
		int[] ranking = employee.getRanking();
		int[] companyRanking = new int[companies.size()];
		int i = 0;
		for (Company c : companies)
			if (c.hasCapacity())
				companyRanking[i++] = ranking[(int) Math.round((ranking.length - 1) * c.getDiversity())];
			else
				companyRanking[i++] = Integer.MAX_VALUE;
		int[] minIndex = minIndex(companyRanking);
		return companies.get(minIndex[(int) Math.floor(minIndex.length * Math.random())]);
	}
	
	/**
	 * Helper method to match. Find the smallest number in an array and return its index.
	 * @param a any array of integers
	 * @return (a randomly selected element from) the argmin of a
	 */
	private int[] minIndex (int[] a) {
		int min = a[0];
		for (int i = 1; i < a.length; i++)
			if (a[i] < min)
				min = a[i];
		int nrMins = 0;
		for (int i = 0; i < a.length; i++)
			if (a[i] == min)
				nrMins++;
		int[] minIndex = new int[nrMins];
		int j = 0;
		for (int i = 0; i < a.length; i++)
			if (a[i] == min)
				minIndex[j++] = i;
		return minIndex;
	}
	
	/**
	 * Perform a simulation run. Matches all agents on the market to the companies on the market.
	 * @return distribution statistics after the run
	 */
	public int[][] run () {
		for (Person employee : employees)
			engage(employee, match(employee));
		return statistics();
	}
	
	/**
	 * Perform a simulation run where agents are sampled randomly from the market with replacement. Sampled agents are matched to companies. For each agent that is added to a company, one random employee is removed from its company, keeping the total number of agents assigned to companies constant.
	 * @param rounds the number of agents to sample in total
	 * @param lag the number of agents to add before one agent is removed for every agent that is added
	 */
	public void runRandom (int rounds, int lag) {
		for (int i = 0; i < rounds; i++) {
			Person employee = employees.get((int) (employees.size() * Math.random()));
			Company company = match(employee);
			// Remove random employee from random company
			if (i >= lag) {
				int[] companySizes = new int[companies.size()];
				int[] cumsum = new int[companySizes.length];
				int j = 0;
				for (Company c : companies) {
					companySizes[j] = c.getSize();
					if (j == 0)
						cumsum[j] = companySizes[j++];
					else
						cumsum[j] = cumsum[j-1] + companySizes[j++];
				}
				int randomEmployee = (int) (cumsum[cumsum.length-1] * Math.random());
				j = -1;
				while (randomEmployee > cumsum[++j]);
				companies.get(j).removeEmployee((int) (companySizes[j] * Math.random()));
			}
			company.addEmployee(employee);
		}
		//return statistics();
	}
	
	/**
	 * Compute the number of employees for each company and the number of these that are minorities.
	 * @return a matrix of companies (rows) vs. number of minority employees (first column) and total number of employees (second column)
	 */
	public int[][] statistics () {
		int[][] statistics = new int[companies.size()][2];
		int i = 0;
		for (Company company : companies) {
			statistics[i][0] = company.getMinorities();
			statistics[i++][1] = company.getSize();
		}
		return statistics;
	}
	
	/**
	 * Compute the dissimilarity coefficient from a matrix of minority vs. total employee statistics.
	 * @param statistics a matrix of statistics as returned by the statistics method
	 * @return the corresponding dissimilarity coefficient
	 */
	public double dissimilarity (int[][] statistics) {
		int[] total = new int[2];
		total[0] = 0;
		total[1] = 0;
		for (int j = 0; j < 2; j++)
			for (int i = 0; i < statistics.length; i++)
				total[j] += statistics[i][j];
		double dissimilarity = 0;
		for (int i = 0; i < statistics.length; i++)
			dissimilarity += 0.5 * Math.abs((1.0 * statistics[i][0]) / total[0] - (1.0 * (statistics[i][1] - statistics[i][0])) / (total[1] - total[0]));
		return dissimilarity;
	}
	
	/**
	 * Pretty print distribution statistics and dissimilarity coefficient.
	 * @param delim1 delimiter to use between statistical measures for each company
	 * @param delim2 delimiter to use between companies
	 * @return distribution statistics and dissimilarity coefficient in readable form
	 */
	public String toString (String delim1, String delim2) {
		int[][] results = statistics();
		String s = dissimilarity(results) + delim2;
		for (int i = 0; i < results.length; i++) {
			for (int j = 0; j < results[i].length; j++)
				s += results[i][j] + delim1;
			s += 100 * results[i][0] / Math.max(1,results[i][1]) + delim2;
		}
		return s;
	}
	
	/**
	 * Formats distribution statistics and dissimilarity coefficient in a readable form. Calls toString(delim1,delim2) with tab and new line.
	 * @return distribution statistics and dissimilarity coefficient in a readable form
	 */
	public String toString () {
		return toString("\t","\n");
	}

}