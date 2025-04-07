def process_streamed_data(self,streamed_data,previous_data,timestamp):
    theta = self.get_view_angle(streamed_data["rotation_w"],streamed_data["rotation_x"],streamed_data["rotation_y"],streamed_data["rotation_z"])
    #streamed_data.update({'time':timestamp})
    streamed_data.update({'theta':theta})
    return streamed_data

def quat_to_euler(self,qw, qx, qy, qz):
    # Yaw (Z-axis rotation)
    yaw = math.atan2(2 * (qx*qy + qw*qz), 1 - 2*(qy**2 + qz**2))
    
    # Pitch (Y-axis rotation)
    sin_pitch = 2 * (qw*qy - qx*qz)
    sin_pitch = max(min(sin_pitch, 1), -1)  # Clamp to valid range
    pitch = math.asin(sin_pitch)
    
    # Roll (X-axis rotation)
    roll = math.atan2(2 * (qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    
    return yaw, pitch, roll

def parse_mocap_skeleton_data(self,mocap_data):
    skeleton_data = mocap_data.get_skeleton_data()
    skeleton_list = skeleton_data.get_skeleton_list()
    skeleton = None
    rigid_body_list = None
    #print(skeleton_data.get_skeleton_count())
    #print(skeleton_list)
    if skeleton_data.get_skeleton_count() == 1:
        skeleton = skeleton_list[0]
        rigid_body_list = []
        for rigid_body in skeleton.get_rigid_body_list():
            #print(rigid_body)
            if rigid_body.is_valid():
                #print('valid')
                important_data = {}
                important_data.update({rigid_body.get_id():[rigid_body.get_position(),rigid_body.get_rotation()]})
                rigid_body_list.append(important_data)
            #else:
                #print(rigid_body.get_as_string())
    else:
        print(f"expected 1 rigid body, got {skeleton_data.get_skeleton_count()}")
    return rigid_body_list

def old_process_streamed_data(self,streamed_data,previous_data,timestamp): # uses MATLAB, kinda large processing overhead...
    streamed_data.update({'time':timestamp})
    #print(streamed_data)
    matlab_array = self.eng.struct(self.dict_to_struct(self.eng,streamed_data))
    if previous_data == None:
        previous_data = self.eng.struct()
    else:
        #print(f"previous_data = {previous_data}")
        previous_data = self.eng.struct(previous_data)
    processed_data = self.eng.process_frame(matlab_array,previous_data,nargout=1)
    processed_dict = dict(processed_data)
    return processed_dict

def dict_to_struct(self,eng, data):
    """
    Safely create a MATLAB struct from a Python dict.
    """
    fields = []
    values = []

    for k, v in data.items():
        fields.append(k)
        if isinstance(v, (int, float)):
            values.append(matlab.double([v]))
        else:
            values.append(v)

    # Now dynamically call struct('field1', value1, 'field2', value2, ...)
    args = []
    for f, v in zip(fields, values):
        args.append(f)
        args.append(v)

    matlab_struct = eng.struct(*args)
    return matlab_struct