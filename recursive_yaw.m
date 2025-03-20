% This script recursively adds features (position velocity, acceleration,
% angular velocity, angular acceleration, yaw angle, yaw rate, and yaw acceleration).

% Load data from CSV
data = readtable('data_first_trial.csv', 'VariableNamingRule', 'preserve');

% Call the function with max 10 iterations (10 seconds of data)
filtered_table = creating_features(data, 1, 10, []);

function filtered_table = creating_features(data, iteration, max_iterations, prev_last_row)
    % This function recursively processes data in chunks of 120 rows at a time
    chunk_size = 120;  % Increased sample rate to 120 per iteration
    
    % If max iterations are reached, stop recursion
    if iteration > max_iterations
        disp("Max iterations reached. Stopping recursion.");
        return;
    end

    % Display current iteration
    disp("Iteration: " + iteration);

    % Process only a chunk of the dataset (120 rows at a time)
    start_idx = (iteration - 1) * chunk_size + 1;
    end_idx = min(iteration * chunk_size, height(data));

    % If no more data is available, stop recursion
    if start_idx > height(data)
        disp("No more data available. Stopping recursion.");
        return;
    end
    
    % Extract the relevant chunk of data
    data_chunk = data(start_idx:end_idx, :);
    
    % Ensure prev_last_row is a table and matches data_chunk structure
    if ~isempty(prev_last_row) && istable(prev_last_row)
        prev_last_row = prev_last_row(:, data_chunk.Properties.VariableNames); % Match column names
        data_chunk = [prev_last_row; data_chunk]; % Concatenate previous last row
    end

    % Extract relevant columns
    time = data_chunk.time;
    rotation_x = data_chunk.rotation_x;
    rotation_y = data_chunk.rotation_y;
    rotation_z = data_chunk.rotation_z;
    rotation_w = data_chunk.rotation_w;
    position_x = data_chunk.position_x;
    position_y = data_chunk.position_y;

    %% **1. Compute Position Velocity and Acceleration**
    if ~isempty(prev_last_row) && istable(prev_last_row)
        prev_time = table2array(prev_last_row(:, "time"));
        prev_position_x = table2array(prev_last_row(:, "position_x"));
        prev_position_y = table2array(prev_last_row(:, "position_y"));

        % Handle missing velocity values from previous iterations
        prev_velocity_x = get_previous_value(prev_last_row, "position_velocity_x", 0);
        prev_velocity_y = get_previous_value(prev_last_row, "position_velocity_y", 0);
    else
        prev_time = time(1);
        prev_position_x = position_x(1);
        prev_position_y = position_y(1);
        prev_velocity_x = 0;
        prev_velocity_y = 0;
    end
    
    % Compute velocity using previous last row
    position_velocity_x = diff([prev_position_x; position_x]) ./ diff([prev_time; time]);
    position_velocity_y = diff([prev_position_y; position_y]) ./ diff([prev_time; time]);

    % Compute acceleration
    position_acceleration_x = diff([prev_velocity_x; position_velocity_x]) ./ diff([prev_time; time]);
    position_acceleration_y = diff([prev_velocity_y; position_velocity_y]) ./ diff([prev_time; time]);

    %% **2. Compute Angular Velocity (ω)**
    if ~isempty(prev_last_row) && istable(prev_last_row)
        prev_rotation_x = table2array(prev_last_row(:, "rotation_x"));
        prev_rotation_y = table2array(prev_last_row(:, "rotation_y"));
        prev_rotation_z = table2array(prev_last_row(:, "rotation_z"));
        prev_rotation_w = table2array(prev_last_row(:, "rotation_w"));
    else
        prev_rotation_x = rotation_x(1);
        prev_rotation_y = rotation_y(1);
        prev_rotation_z = rotation_z(1);
        prev_rotation_w = rotation_w(1);
    end

    % Compute quaternion derivatives
    dq_x = diff([prev_rotation_x; rotation_x]) ./ diff([prev_time; time]);
    dq_y = diff([prev_rotation_y; rotation_y]) ./ diff([prev_time; time]);
    dq_z = diff([prev_rotation_z; rotation_z]) ./ diff([prev_time; time]);
    dq_w = diff([prev_rotation_w; rotation_w]) ./ diff([prev_time; time]);

    % Compute angular velocity
    omega_x = 2 * (rotation_w .* dq_x + rotation_x .* dq_w);
    omega_y = 2 * (rotation_w .* dq_y + rotation_y .* dq_w);
    omega_z = 2 * (rotation_w .* dq_z + rotation_z .* dq_w);

    %% **3. Compute Angular Acceleration (α)**
    min_size_alpha = min(length(omega_x)-1, length(time)-1);
    alpha_x = [diff(omega_x(1:min_size_alpha)) ./ diff(time(1:min_size_alpha)); NaN];
    alpha_y = [diff(omega_y(1:min_size_alpha)) ./ diff(time(1:min_size_alpha)); NaN];
    alpha_z = [diff(omega_z(1:min_size_alpha)) ./ diff(time(1:min_size_alpha)); NaN];

    %% **4. Compute Yaw Angle (θ) and Convert to Degrees**
    yaw_angle = atan2(2 * (rotation_w .* rotation_z + rotation_x .* rotation_y), ...
                      1 - 2 * (rotation_y.^2 + rotation_z.^2));
    yaw_angle_deg = rad2deg(yaw_angle); % Convert to degrees

    % Compute yaw rate (dθ/dt)
    min_size_yaw = min(length(yaw_angle)-1, length(time)-1);
    yaw_rate = [diff(yaw_angle(1:min_size_yaw)) ./ diff(time(1:min_size_yaw)); NaN];

    % Compute yaw acceleration (d²θ/dt²)
    min_size_yaw_acc = min(length(yaw_rate)-1, length(time)-2);
    yaw_acceleration = [diff(yaw_rate(1:min_size_yaw_acc)) ./ diff(time(1:min_size_yaw_acc)); NaN];

    %% **5. Store Results in the Table**
    num_rows = height(data_chunk); % Ensure consistent row count

    data_chunk.dq_x = pad_array(dq_x, num_rows);
    data_chunk.dq_y = pad_array(dq_y, num_rows);
    data_chunk.dq_z = pad_array(dq_z, num_rows);
    data_chunk.dq_w = pad_array(dq_w, num_rows);

    data_chunk.omega_x = pad_array(omega_x, num_rows);
    data_chunk.omega_y = pad_array(omega_y, num_rows);
    data_chunk.omega_z = pad_array(omega_z, num_rows);

    data_chunk.alpha_x = pad_array(alpha_x, num_rows);
    data_chunk.alpha_y = pad_array(alpha_y, num_rows);
    data_chunk.alpha_z = pad_array(alpha_z, num_rows);

    data_chunk.yaw_angle_deg = yaw_angle_deg;
    data_chunk.yaw_rate = pad_array(yaw_rate, num_rows);
    data_chunk.yaw_acceleration = pad_array(yaw_acceleration, num_rows);

    %% **6. Recursive Call**
    prev_last_row = data_chunk(end, :);
    if iteration < max_iterations
        next_table = creating_features(data, iteration + 1, max_iterations, prev_last_row);
        filtered_table = [data_chunk; next_table];
    else
        filtered_table = data_chunk;
    end
end

function padded_array = pad_array(arr, desired_length)
    arr = arr(:);
    padding_size = desired_length - length(arr);
    padded_array = [arr; NaN(padding_size, 1)];
end

function value = get_previous_value(prev_last_row, field_name, default_value)
    if ismember(field_name, prev_last_row.Properties.VariableNames)
        value = table2array(prev_last_row(:, field_name));
    else
        value = default_value;
    end
end
