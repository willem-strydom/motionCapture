% This script is used in order to add features for the specific data

% Load the filtered CSV file
data = readtable('filtered_rigid_body_data_with_time.csv', 'VariableNamingRule', 'preserve');

% Display actual column names to verify
disp("Column Names:");
disp(data.Properties.VariableNames);

% Extract relevant columns using MATLAB's assigned names
time = data.time;
rotation_x = data.rotation_x;
rotation_y = data.rotation_y;
rotation_z = data.rotation_z;
rotation_w = data.rotation_w;
position_x = data.position_x;
position_y = data.position_y;

%% **1. Compute Yaw Angle (Rotation Around Y-Axis)**
yaw_angle = atan2(2 * (rotation_w .* rotation_y + rotation_x .* rotation_z), ...
                 1 - 2 * (rotation_y.^2 + rotation_z.^2));

% Convert yaw angle from radians to degrees
theta = yaw_angle * (180 / pi);

%% **2. Compute Position Velocity and Acceleration**
% Calculate position velocity using (X1 - X0) / T1
position_velocity_x = (position_x(2:end) - position_x(1:end-1)) ./ time(2:end);
position_velocity_y = (position_y(2:end) - position_y(1:end-1)) ./ time(2:end);

% Calculate position acceleration using (V1 - V0) / T1
position_acceleration_x = (position_velocity_x(2:end) - position_velocity_x(1:end-1)) ./ time(3:end);
position_acceleration_y = (position_velocity_y(2:end) - position_velocity_y(1:end-1)) ./ time(3:end);

% Ensure matching row sizes by padding with 0 instead of NaN
position_velocity_x = [0; position_velocity_x]; % First entry is 0
position_velocity_y = [0; position_velocity_y];
position_acceleration_x = [0; 0; position_acceleration_x]; % First two entries are 0
position_acceleration_y = [0; 0; position_acceleration_y];

%% **3. Compute Angular Velocity (ω)**
dq_x = diff(rotation_x) ./ diff(time);
dq_y = diff(rotation_y) ./ diff(time);
dq_z = diff(rotation_z) ./ diff(time);
dq_w = diff(rotation_w) ./ diff(time);

% Compute angular velocity components using ω = 2 * (q_w * dq + q × dq)
omega_x = 2 * (rotation_w(2:end) .* dq_x + rotation_x(2:end) .* dq_w);
omega_y = 2 * (rotation_w(2:end) .* dq_y + rotation_y(2:end) .* dq_w);
omega_z = 2 * (rotation_w(2:end) .* dq_z + rotation_z(2:end) .* dq_w);

% Ensure proper dimensions by padding first value with 0
omega_x = [0; omega_x];
omega_y = [0; omega_y];
omega_z = [0; omega_z];

%% **4. Compute Angular Acceleration (α)**
alpha_x = diff(omega_x) ./ diff(time); % Ensure both arrays are size N-1
alpha_y = diff(omega_y) ./ diff(time);
alpha_z = diff(omega_z) ./ diff(time);

% Padding with 0 to match the table row size
alpha_x = [0; alpha_x]; 
alpha_y = [0; alpha_y];
alpha_z = [0; alpha_z];

%% **5. Predict Future Quaternion Rotation Using Quaternion Integration**
dt = mean(diff(time)); % Approximate time step
predicted_quaternion = zeros(length(rotation_w)-1, 4); % Placeholder for predictions

for i = 1:length(rotation_w)-1
    q_t = [rotation_w(i), rotation_x(i), rotation_y(i), rotation_z(i)]; % Current quaternion
    omega_quat = [0, omega_x(i), omega_y(i), omega_z(i)]; % Angular velocity as quaternion
    
    % Compute quaternion derivative using dq = 0.5 * q_t * omega * dt
    dq = 0.5 * quatmultiply(q_t, omega_quat) * dt; 
    
    % Integrate to get next quaternion
    predicted_quaternion(i, :) = q_t + dq;
end

% Pad the predicted quaternion to match the table size
predicted_quaternion = [predicted_quaternion; [0, 0, 0, 0]]; 

%% **6. Create a new filtered table with all computed features**
filtered_table = table(time, rotation_x, rotation_y, rotation_z, rotation_w, ...
                       position_x, position_y, position_velocity_x, position_velocity_y, ...
                       position_acceleration_x, position_acceleration_y, ...
                       omega_x, omega_y, omega_z, alpha_x, alpha_y, alpha_z, ...
                       predicted_quaternion(:,1), predicted_quaternion(:,2), ...
                       predicted_quaternion(:,3), predicted_quaternion(:,4), yaw_angle, theta, ...
                       'VariableNames', {'time', 'rotation_x', 'rotation_y', 'rotation_z', 'rotation_w', ...
                                         'position_x', 'position_y', 'position_velocity_x', 'position_velocity_y', ...
                                         'position_acceleration_x', 'position_acceleration_y', ...
                                         'omega_x', 'omega_y', 'omega_z', 'alpha_x', 'alpha_y', 'alpha_z', ...
                                         'pred_w', 'pred_x', 'pred_y', 'pred_z', 'yaw_angle', 'theta'});

%% **7. Display First Few Entries**
num_rows = 5; % Number of rows to display

disp("Displaying First Few Entries of Filtered Table:");
disp(filtered_table(1:num_rows, :));
