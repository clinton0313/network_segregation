import java.util.List;
import java.util.ArrayList;

/**
 * A company is a container for agents, termed employees. A company also has its number of minorities a property, and an optional limited static capacity (same for all companies).
 */
public class Company {

	private static int capacity = Integer.MAX_VALUE;
	private List<Person> employees;
	private int minorities;
	
	/**
	 * Creates a company with no employees.
	 */
	public Company () {
		employees = new ArrayList<Person>();
		minorities = 0;
	}
	
	/**
	 * Limit the capacity of all companies.
	 */
	public static void setCapacity (int capacity) {
		Company.capacity = capacity;
	}
	
	/**
	 * Add one employee to the company.
	 * @param employee the agent to add
	 * @return whether employee could be added (within capacity constraint)
	 */
	public boolean addEmployee (Person employee) {
		if (employees.size() >= capacity)
			return false;
		employees.add(employee);
		if (employee.isMinority())
			minorities++;
		return true;
	}
	
	/**
	 * Remove a specified employee from the company.
	 * @param employee the agent to remove
	 * @return whether employee existed and could be removed
	 */
	public boolean removeEmployee (Person employee) {
		if (!employees.remove(employee))
			return false;
		if (employee.isMinority())
			minorities--;
		return true;
	}
	
	/**
	 * Remove the ith oldest employee from the company.
	 * @param i the number of the agent to remove
	 * @return whether company had at least i employees
	 */
	public boolean removeEmployee (int i) {
		if (i >= employees.size())
			return false;
		Person employee = employees.remove(i);
		if (employee.isMinority())
			minorities--;
		return true;		
	}
	
	/**
	 * @return number of employees
	 */
	public int getSize () {
		return employees.size();
	}
	
	/**
	 * @return number of minority employees
	 */
	public int getMinorities () {
		return minorities;
	}
	
	/**
	 * @return fraction of minority employees
	 */
	public double getDiversity () {
		if (employees.size() > 0)
			return 1.0 * minorities / employees.size();
		else
			return 0;
	}
	
	/**
	 * @return whether company has capacity for at least one more employee
	 */
	public boolean hasCapacity () {
		return employees.size() < capacity;
	}

}