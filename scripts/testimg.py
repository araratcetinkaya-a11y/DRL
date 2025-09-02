import inspect, airsim

print("airsim file:", airsim.__file__)
print("simGetImages signature:", inspect.signature(airsim.client.VehicleClient.simGetImages))
