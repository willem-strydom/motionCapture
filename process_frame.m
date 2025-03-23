function transformed_row = process_frame(current_row, prev_output)
    % Define machine centers (x, z)
    machine_centers = [
        1, 1;   % Machine 1
        2, 2;   % Machine 2
        3, 3;   % Machine 3
        4, 4    % Machine 4
    ];

    % Convert struct to table if needed
    if isstruct(current_row)
        current_row = struct2table(current_row);
    end
    if isstruct(prev_output)
        prev_output = struct2table(prev_output);
    end

    % Extract current values
    time = current_row.time;
    position_x = current_row.position_x;
    position_y = current_row.position_y;
    position_z = current_row.position_z;
    rotation_x = current_row.rotation_x;
    rotation_y = current_row.rotation_y;
    rotation_z = current_row.rotation_z;
    rotation_w = current_row.rotation_w;

    % Compute yaw angle (theta)
    theta = rad2deg(atan2(2 * (rotation_w * rotation_z + rotation_x * rotation_y), ...
                          1 - 2 * (rotation_y^2 + rotation_z^2)));

    % Initialize defaults
    velocity_x = NaN; acceleration_x = NaN;
    velocity_z = NaN; acceleration_z = NaN;
    theta_dot = NaN; theta_dot_dot = NaN;

    % Compute velocities and accelerations if previous frame exists
    if ~isempty(prev_output)
        dt = time - prev_output.time;

        if isfield(prev_output, 'position_x') && isfield(prev_output, 'position_z')
            velocity_x = (position_x - prev_output.position_x) / dt;
            velocity_z = (position_z - prev_output.position_z) / dt;
        end

        if isfield(prev_output, 'velocity_x') && ~isnan(velocity_x)
            acceleration_x = (velocity_x - prev_output.velocity_x) / dt;
        end
        if isfield(prev_output, 'velocity_z') && ~isnan(velocity_z)
            acceleration_z = (velocity_z - prev_output.velocity_z) / dt;
        end

        if isfield(prev_output, 'theta')
            theta_dot = (theta - prev_output.theta) / dt;
        end
        if isfield(prev_output, 'theta_dot') && ~isnan(theta_dot)
            theta_dot_dot = (theta_dot - prev_output.theta_dot) / dt;
        end
    end

    % Compute theta_1 to theta_4 (angle between yaw direction and machine vector)
    theta_vec = [cosd(theta), sind(theta)];
    theta_n = zeros(1, 4);

    for i = 1:4
        dx = machine_centers(i, 1) - position_x;
        dz = machine_centers(i, 2) - position_z;
        machine_vec = [dx, dz];
        norm_product = norm(theta_vec) * norm(machine_vec);
        if norm_product > 0
            angle_rad = acos(dot(theta_vec, machine_vec) / norm_product);
            theta_n(i) = rad2deg(angle_rad);
        else
            theta_n(i) = 0;
        end
    end

    % Return all processed values
    transformed_row = struct(...
        'time', time, ...
        'position_x', position_x, ...
        'position_y', position_y, ...
        'position_z', position_z, ...
        'rotation_x', rotation_x, ...
        'rotation_y', rotation_y, ...
        'rotation_z', rotation_z, ...
        'rotation_w', rotation_w, ...
        'velocity_x', default0(velocity_x), ...
        'acceleration_x', default0(acceleration_x), ...
        'velocity_z', default0(velocity_z), ...
        'acceleration_z', default0(acceleration_z), ...
        'theta', theta, ...
        'theta_dot', default0(theta_dot), ...
        'theta_dot_dot', default0(theta_dot_dot), ...
        'theta_1', theta_n(1), ...
        'theta_2', theta_n(2), ...
        'theta_3', theta_n(3), ...
        'theta_4', theta_n(4) ...
    );
end

function val = default0(x)
    if isnan(x)
        val = 0;
    else
        val = x;
    end
end
