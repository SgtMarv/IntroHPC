#! /home/max/.rvm/rubies/ruby-1.9.2-p290/bin/ruby

lines = File.open(ARGV[0]).readlines

avg = 0.0
lines.each do |line|
    avg += line.to_f
end
avg = (avg/lines.count)

err = 0.0
lines.each do |line|
    err += (line.to_f-avg)**2
end

err = Math.sqrt((1/(lines.count-1.0))*err)

puts avg.to_s + " " + err.to_s + "\n"

