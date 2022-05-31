import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * The main class of the workplace segregation model. The model uses agents' preference rankings of companies w.r.t. proportion of minority group employees and allocates agents to companies accordingly. The output is the number of employees and the number of which are minorities at each company, and corresponding dissimilarity coefficient. Run the program with the flag -h for options and more instructions on running simulations.
 */
public class Workplaces {

	/**
	 * Parse arguments as flags and parameters for the model, and run simulations based on this setup. Results are printed to screen and/or file. Run with flag -h for instructions on how to write input and what parameters are available.
	 * @param args flags and arguments for the simulation model
	 * @throws IOException if input or output file cannot be read or written to/created
	 */
	public static void main (String[] args) throws IOException {
		
		String input = "", output = "";
		int nrCompanies = 5, rounds = 1000000, lag = 1000, capacity = Integer.MAX_VALUE, repeat = 1;
		int[] minoritiesAndMajorities = new int[0];
		boolean hide = false;
		
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "-f":
				case "-file":
					input = args[++i];
					break;
				case "-nr":
					nrCompanies = Integer.parseInt(args[++i]);
					break;
				case "-t":
					rounds = Integer.parseInt(args[++i]);
					break;
				case "-r":
				case "-rep":
				case "-repeat":
					repeat = Integer.parseInt(args[++i]);
					break;
				case "-n":
					lag = Integer.parseInt(args[++i]);
					break;
				case "-c":
				case "-cap":
				case "-capacity":
					capacity = Integer.parseInt(args[++i]);
					break;
				case "-i":
				case "-init":
				case "-initialise":
					int l = 0;
					while (i + l + 1 < args.length && !args[i+l+1].startsWith("-"))
						l++;
					if (l % 2 != 0) {
						System.err.println("Option -init requires an even number of integers");
						System.exit(1);
					}
					minoritiesAndMajorities = new int[l];
					for (int j = 0; j < l; j++)
						minoritiesAndMajorities[j] = Integer.parseInt(args[i+j+1]);
					i += l;
					break;
				case "-o":
				case "-out":
				case "-output":
					output = args[++i];
					break;
				case "-hide":
					hide = true;
					break;
				case "-h":
				case "-help":
					System.out.println("Usage: java Workplaces -option [argument(s)]");
					System.out.println("-f(ile)\t CSV file with each agent's minority status (1/0) & preference ranking");
					System.out.println("-nr \t Number of companies (" + nrCompanies + ")");
					System.out.println("-t \t Number of rounds (" + rounds + ")");
					System.out.println("-r(ep) \t Number of repetitions (" + repeat + ")");
					System.out.println("-n \t Number of agents (" + lag + ")");
					System.out.println("-c(ap)\t Number of employees each company has capacity for (" + capacity + ")");
					System.out.println("-i(nit)\t Vector of minorities and majorities initially per company (0 0 ... 0)");
					System.out.println("-o(ut)\t CSV file to write output to");
					System.out.println("-hide\t Do not print results on screen");
					System.exit(0);
					break;
				default:
					System.err.println("Input is given in the wrong format. Use -h for help");
					System.exit(1);
			}
		}
		
		List<Person> employees = new ArrayList<Person>();
		List<String> lines = Files.readAllLines(Paths.get(input + ".csv"), StandardCharsets.UTF_8);
		for (String line : lines) {
			String[] fields = line.split(",|;");
			boolean minority = false;
			if (Integer.parseInt(fields[0]) == 1)
				minority = true;
			int[] ranking = new int[fields.length-1];
			for (int i = 0; i < ranking.length; i++)
				ranking[i] = Integer.parseInt(fields[i+1]);
			employees.add(new Person(minority, ranking));
		}
		
		Market[] markets = new Market[repeat];
		for (int i = 0; i < repeat; i++) {		
			markets[i] = new Market(employees, nrCompanies, capacity);
			if (minoritiesAndMajorities.length > 0)
				initialise(markets[i], minoritiesAndMajorities);
			markets[i].runRandom(rounds,lag);
			if (!hide)
				System.out.println(markets[i]);
		}
		
		if (output != "") {
			File f = new File(output + ".csv");
			boolean newFile = !f.exists();
			FileWriter w = new FileWriter(f, true);
			if (newFile)
				w.write("file;nr;t;n;capacity;d;minorities;employees;ratio\n");
			for (int i = 0; i < repeat; i++)
				w.append(input + ";" + nrCompanies + ";" + rounds + ";" + lag + ";" + capacity + ";" + markets[i].toString(";", ";") + "\n");
			w.flush();
			w.close();
		}
		
	}
	
	/**
	 * Helper method to main. Initialise a job market with prespecified number of minorities and majorities at each company.
	 * @param market the market for which agents will be initially allocated
	 * @param minoritiesAndMajorities the number of minorities and majorities, respectively, to allocate initially for each company, with two entries for each company
	 */
	private static void initialise (Market market, int[] minoritiesAndMajorities) {
		boolean minority = true;
		int k = 0;
		for (int nrPeople : minoritiesAndMajorities) {
			for (int i = 0; i < nrPeople; i++)
				market.engage(new Person(minority), k);
			minority = !minority;
			if (minority)
				k++;
		}
	}
	
}