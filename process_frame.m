function transformed_row = process_frame(current_row, prev_row)
    % current_row and prev_row are assumed to be tables with the same structure
    % If prev_row is empty, use the current row values as the base.
    if isstruct(current_row)
        current_row = struct2table(current_row);
    end
    if isstruct(prev_row)
        prev_row = struct2table(prev_row);
    end
    % Extract values from the current row
    time = current_row.time;
    rotation_x = current_row.rotation_x;
    rotation_y = current_row.rotation_y;
    rotation_z = current_row.rotation_z;
    rotation_w = current_row.rotation_w;
    position_x = current_row.position_x;
    position_z = current_row.position_z;
    
    % Use previous row values if available
    if ~isempty(prev_row)
        prev_time = prev_row.time;
        prev_position_x = prev_row.position_x;
        prev_position_z = prev_row.position_z;
        % Compute velocity from previous and current rows
        velocity_x = (position_x - prev_position_x) / (time - prev_time);
        velocity_y = (position_z - prev_position_z) / (time - prev_time);
    else
        velocity_x = 0;
        velocity_y = 0;
    end

    % Compute quaternion derivative using a similar approach
    if ~isempty(prev_row)
        prev_rotation_x = prev_row.rotation_x;
        prev_rotation_y = prev_row.rotation_y;
        prev_rotation_z = prev_row.rotation_z;
        prev_rotation_w = prev_row.rotation_w;
        dt = time - prev_time;
        dq_x = (rotation_x - prev_rotation_x) / dt;
        dq_y = (rotation_y - prev_rotation_y) / dt;
        dq_z = (rotation_z - prev_rotation_z) / dt;
        dq_w = (rotation_w - prev_rotation_w) / dt;
    else
        dq_x = 0; dq_y = 0; dq_z = 0; dq_w = 0;
    end

    % Compute angular velocity (using simplified calculations for one sample)
    omega_x = 2 * (rotation_w * dq_x + rotation_x * dq_w);
    omega_y = 2 * (rotation_w * dq_y + rotation_y * dq_w);
    omega_z = 2 * (rotation_w * dq_z + rotation_z * dq_w);

    % Compute yaw angle (θ) and convert to degrees
    yaw_angle = atan2(2 * (rotation_w * rotation_z + rotation_x * rotation_y), ...
                      1 - 2 * (rotation_y^2 + rotation_z^2));
    yaw_angle_deg = rad2deg(yaw_angle);

    % Return the processed data in a table
    transformed_row = struct(...
    'position_x', position_x, ...
    'position_y', current_row.position_y, ...
    'position_z', position_z, ...
    'rotation_x', rotation_x, ...
    'rotation_y', rotation_y, ...
    'rotation_z', rotation_z, ...
    'rotation_w', rotation_w, ...
    'time', time, ...
    'velocity_x', velocity_x, ...
    'velocity_y', velocity_y, ...
    'dq_x', dq_x, ...
    'dq_y', dq_y, ...
    'dq_z', dq_z, ...
    'dq_w', dq_w, ...
    'omega_x', omega_x, ...
    'omega_y', omega_y, ...
    'omega_z', omega_z, ...
    'yaw_angle_deg', yaw_angle_deg);

end
