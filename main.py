from Agent import Agent

agent = Agent()
agent.init()

agent.generate_images(15)

# agent.gui.place_dot(88.5694, 47.1211)

# dots = agent.gui.dots

agent.gui.generate_random_output()

dots = []
# dots.append((-88.5694, 47.1211))
# dots.append((-71.0589, 42.3601))

for city_loc in agent.city_images:
    dots.append(city_loc.get_loc())

agent.gui.show(dots=dots, display_coords=False)
