# import library
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# importing data set
df = pd.read_csv('Spatial_Navigation_Research_DATA_2012-2013[NO_NAMES]_62sketchmappers_USETHIS-2.csv')
print(df)

X = df.iloc[:, 1:45].values
y = df.iloc[:, 0].values

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Print the length of the array
print(f'\nThe length of the y_train array is: {len(y_train)}')
# Print the length of the array
print(f'The length of the y_test array is: {len(y_test)}')

d_train = pd.read_csv('train_Set_Output_LDA.csv')

x_train_d1 = d_train.iloc[:, 1].values
x_train_d2 = d_train.iloc[:, 2].values
# Print the length of the array
print(f'The length of the x_train_d1 array is: {len(x_train_d1)}')
# Print the length of the array
print(f'The length of the x_train_d2 array is: {len(x_train_d2)}')

# IPDT_ACC_train against LDA_x_train_dimension1
# Perform linear regression to get the best-fit line
# Add a constant term to the predictor variable
X_with_const = sm.add_constant(x_train_d1)

# Create and fit the linear regression model using statsmodels
model = sm.OLS(y_train, X_with_const).fit()

print("\n(1)")
# Print the summary which includes p-values
print(model.summary())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Print coefficients and p-values
for i in range(len(coefficients)):
    print(f'Coefficient {i}: {coefficients[i]:.4f}  \t P-value: {p_values[i]:.4f}')

# Extract coefficients
intercept = model.params[0]
slope = model.params[1]
print(f'equation: y = {intercept:.2f} + {slope:.2f}x\n' )

# Plot the regression line
plt.scatter(x_train_d1, y_train, color='darkviolet')
plt.plot(x_train_d1, intercept + slope * x_train_d1, color='red', label='Best Fit Line')
plt.title("IPDT_ACC_train against LDA_x_train_dimension1")
plt.xlabel('LDA')
plt.ylabel('IPDT_ACC')
plt.show()


# IPDT_ACC_train against LDA_x_train_dimension2
# Perform linear regression to get the best-fit line
# Add a constant term to the predictor variable
X_with_const = sm.add_constant(x_train_d2)

# Create and fit the linear regression model using statsmodels
model = sm.OLS(y_train, X_with_const).fit()

print("(2)")
# Print the summary which includes p-values
print(model.summary())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Print coefficients and p-values
for i in range(len(coefficients)):
    print(f'Coefficient {i}: {coefficients[i]:.4f}  \t P-value: {p_values[i]:.4f}')

# Extract coefficients
intercept = model.params[0]
slope = model.params[1]
print(f'equation: y = {intercept:.2f} + {slope:.2f}x\n' )

# Plot the regression line
plt.scatter(x_train_d2, y_train, color='blue')
plt.plot(x_train_d2, intercept + slope * x_train_d2, color='red', label='Best Fit Line')
plt.title("IPDT_ACC_train against LDA_x_train_dimension2")
plt.xlabel('LDA')
plt.ylabel('IPDT_ACC')
plt.show()


# plot the test_output of LDA
d_train = pd.read_csv('test_Set_Output_LDA.csv')

x_test_d1 = d_train.iloc[:, 1].values
x_test_d2 = d_train.iloc[:, 2].values

# Print the length of the array
print(f'The length of the x_test_d1 array is: {len(x_test_d1)}')
# Print the length of the array
print(f'The length of the x_test_d2 array is: {len(x_test_d2)}')

# IPDT_ACC_test against LDA_x_test_dimension1
# Perform linear regression to get the best-fit line
# Add a constant term to the predictor variable
X_with_const = sm.add_constant(x_test_d1)

# Create and fit the linear regression model using statsmodels
model = sm.OLS(y_test, X_with_const).fit()

print("(3)")
# Print the summary which includes p-values
print(model.summary())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Print coefficients and p-values
for i in range(len(coefficients)):
    print(f'Coefficient {i}: {coefficients[i]:.4f}  \t P-value: {p_values[i]:.4f}')

# Extract coefficients
intercept = model.params[0]
slope = model.params[1]
print(f'equation: y = {intercept:.2f} + {slope:.2f}x\n' )

# Plot the regression line
plt.scatter(x_test_d1, y_test, color='green')
plt.plot(x_test_d1, intercept + slope * x_test_d1, color='red', label='Best Fit Line')
plt.title("IPDT_ACC_test against LDA_x_test_dimension1")
plt.xlabel('LDA')
plt.ylabel('IPDT_ACC')
plt.show()


# # IPDT_ACC_test against LDA_x_test_dimension2
# # Perform linear regression to get the best-fit line
# Add a constant term to the predictor variable
X_with_const = sm.add_constant(x_test_d2)

# Create and fit the linear regression model using statsmodels
model = sm.OLS(y_test, X_with_const).fit()

print("(4)")
# Print the summary which includes p-values
print(model.summary())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Print coefficients and p-values
for i in range(len(coefficients)):
    print(f'Coefficient {i}: {coefficients[i]:.4f}  \t P-value: {p_values[i]:.4f}')

# Extract coefficients
intercept = model.params[0]
slope = model.params[1]
print(f'equation: y = {intercept:.2f} + {slope:.2f}x\n' )

# Plot the regression line
plt.scatter(x_test_d2, y_test, color='orange')
plt.plot(x_test_d2, intercept + slope * x_test_d2, color='red', label='Best Fit Line')
plt.title("IPDT_ACC_test against LDA_x_test_dimension2")
plt.xlabel('LDA')
plt.ylabel('IPDT_ACC')
plt.show()






